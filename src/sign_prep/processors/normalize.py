"""Normalization processor: visibility masking + clip normalization."""

import glob
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from .base import BaseProcessor
from ..presets import resolve_keypoint_indices
from ..registry import register_processor


def _load_clip(path: str) -> np.ndarray:
    """Load raw landmark clip from .npy file. Returns shape (T, K, 4)."""
    arr = np.load(path).astype(np.float32)

    if arr.ndim == 2:
        T, F = arr.shape
        if F % 4 == 0:
            K = F // 4
            if K in [85, 133, 543, 553]:
                arr = arr.reshape(T, K, 4)
            else:
                raise ValueError(f"Unsupported shape {arr.shape} in {path}")
        else:
            raise ValueError(f"Invalid feature count {F} in {path}")
    elif arr.ndim == 3:
        T, K, C = arr.shape
        if C != 4:
            raise ValueError(f"Expected 4 channels, got {C} in {path}")
    else:
        raise ValueError(f"Unexpected ndim={arr.ndim} for {path}")

    return arr


def _apply_keypoint_reduction(
    clip_xyzv: np.ndarray,
    keypoint_indices: List[int],
) -> np.ndarray:
    """Reduce keypoints by selecting specific indices."""
    if clip_xyzv.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {clip_xyzv.shape}")
    if not keypoint_indices:
        raise ValueError("keypoint_indices is empty — cannot reduce to zero keypoints")
    T, K, C = clip_xyzv.shape
    if max(keypoint_indices) >= K:
        raise ValueError(
            f"Index {max(keypoint_indices)} out of range for {K} keypoints. "
            f"Check that keypoint_preset matches the extractor output."
        )
    return clip_xyzv[:, keypoint_indices, :]


def _apply_visibility_mask(
    clip_xyzv: np.ndarray,
    mask_empty_frames: bool,
    mask_low_confidence: bool,
    visibility_threshold: float,
    missing_value: float,
) -> np.ndarray:
    """Apply visibility masking. Returns (T, K, 3) with sentinels."""
    T, K, _ = clip_xyzv.shape
    xyz = clip_xyzv[..., :3].copy()
    vis = clip_xyzv[..., 3]

    if mask_empty_frames:
        frame_all_zero = np.all(clip_xyzv == 0.0, axis=(1, 2))
        for t in range(T):
            if frame_all_zero[t]:
                xyz[t, :, :] = missing_value

    if mask_low_confidence:
        low_vis = vis < visibility_threshold
        all_zero = np.all(clip_xyzv == 0.0, axis=-1)
        missing = np.logical_or(low_vis, all_zero)
        xyz[missing] = missing_value

    return xyz


def _normalize_clip_xyz(
    xyz_masked: np.ndarray,
    mode: str,
    missing_value: float,
) -> np.ndarray:
    """Normalize landmarks using whole-clip scaling."""
    import logging
    logger = logging.getLogger("sign_prep.normalize")

    if xyz_masked.ndim != 3 or xyz_masked.shape[-1] != 3:
        raise ValueError(f"Expected shape (T, K, 3), got {xyz_masked.shape}")

    out = xyz_masked.copy()
    invalid = xyz_masked == missing_value
    valid_points_mask = ~np.any(invalid, axis=-1)

    if not np.any(valid_points_mask):
        return out

    valid_coords = xyz_masked[valid_points_mask]
    x, y, z = valid_coords[:, 0], valid_coords[:, 1], valid_coords[:, 2]

    if mode == "isotropic_3d":
        coord_min = valid_coords.min(axis=0)
        coord_max = valid_coords.max(axis=0)
        coord_range = coord_max - coord_min
        max_range = float(np.max(coord_range))

        if np.isclose(max_range, 0.0):
            scaled_valid = np.zeros_like(valid_coords, dtype=np.float32)
        else:
            scaled_valid = ((valid_coords - coord_min) / max_range).astype(np.float32)

    elif mode == "xy_isotropic_z_minmax":
        x_min, x_max = float(x.min()), float(x.max())
        y_min, y_max = float(y.min()), float(y.max())
        max_xy = max(x_max - x_min, y_max - y_min)

        if np.isclose(max_xy, 0.0):
            x_s = np.zeros_like(x, dtype=np.float32)
            y_s = np.zeros_like(y, dtype=np.float32)
        else:
            x_s = (x - x_min) / max_xy
            y_s = (y - y_min) / max_xy

        z_min, z_max = float(z.min()), float(z.max())
        z_range = z_max - z_min
        z_s = (
            np.zeros_like(z, dtype=np.float32) if np.isclose(z_range, 0.0)
            else (z - z_min) / z_range
        )

        scaled_valid = np.stack([x_s, y_s, z_s], axis=-1).astype(np.float32)
    else:
        raise ValueError(f"Unknown mode: '{mode}'")

    out[valid_points_mask] = scaled_valid
    return out


def _process_single_file(args) -> Tuple[str, bool, str]:
    """Process a single landmark file through the normalization pipeline."""
    input_path, output_path, normalize_config = args
    filename = os.path.basename(input_path)

    try:
        if normalize_config["skip_existing"] and os.path.exists(output_path):
            return filename, True, "skipped (exists)"

        clip_raw = _load_clip(input_path)
        T, K, C = clip_raw.shape

        # Keypoint reduction
        if normalize_config["select_keypoints"]:
            indices = resolve_keypoint_indices(
                normalize_config.get("keypoint_preset"),
                normalize_config.get("keypoint_indices"),
            )
            if indices is not None:
                clip_raw = _apply_keypoint_reduction(clip_raw, indices)

        # Visibility masking
        xyz_masked = _apply_visibility_mask(
            clip_raw,
            normalize_config["mask_empty_frames"],
            normalize_config["mask_low_confidence"],
            normalize_config["visibility_threshold"],
            normalize_config["missing_value"],
        )

        # Normalization
        xyz_norm = _normalize_clip_xyz(
            xyz_masked, normalize_config["mode"], normalize_config["missing_value"]
        )

        # Optionally drop z
        if normalize_config["remove_z"]:
            xyz_norm = xyz_norm[..., :2]

        # Flatten
        T2, K2, C_out = xyz_norm.shape
        flattened = xyz_norm.reshape(T2, K2 * C_out).astype(np.float32)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, flattened)

        return filename, True, ""
    except Exception as e:
        return filename, False, str(e)


@register_processor("normalize")
class NormalizeProcessor(BaseProcessor):
    name = "normalize"

    def run(self, context):
        cfg = self.config
        input_dir = cfg.paths.landmarks
        output_dir = cfg.paths.normalized

        os.makedirs(output_dir, exist_ok=True)

        pattern = os.path.join(input_dir, "**", "*.npy")
        npy_files = glob.glob(pattern, recursive=True)

        if not npy_files:
            self.logger.warning("No .npy files found in %s", input_dir)
            context.stats["normalize"] = {"total": 0}
            return context

        normalize_config = {
            "mode": cfg.normalize.mode,
            "remove_z": cfg.normalize.remove_z,
            "select_keypoints": cfg.normalize.select_keypoints,
            "keypoint_preset": cfg.normalize.keypoint_preset,
            "keypoint_indices": cfg.normalize.keypoint_indices,
            "mask_empty_frames": cfg.normalize.mask_empty_frames,
            "mask_low_confidence": cfg.normalize.mask_low_confidence,
            "visibility_threshold": cfg.normalize.visibility_threshold,
            "missing_value": cfg.normalize.missing_value,
            "skip_existing": cfg.processing.skip_existing,
        }

        tasks = []
        for inp in npy_files:
            rel = os.path.relpath(inp, input_dir)
            out = os.path.join(output_dir, rel)
            tasks.append((inp, out, normalize_config))

        self.logger.info("Normalizing %d files from %s", len(tasks), input_dir)

        success = skip = errors = 0
        with ProcessPoolExecutor(max_workers=cfg.processing.max_workers) as executor:
            futures = {
                executor.submit(_process_single_file, t): t for t in tasks
            }
            with tqdm(total=len(tasks), desc="Normalizing") as pbar:
                for future in as_completed(futures):
                    fname, ok, msg = future.result()
                    if ok:
                        if msg == "skipped (exists)":
                            skip += 1
                        else:
                            success += 1
                    else:
                        errors += 1
                        self.logger.error("Failed: %s - %s", fname, msg)
                    pbar.update(1)

        context.stats["normalize"] = {
            "total": len(tasks),
            "success": success,
            "skipped": skip,
            "errors": errors,
        }
        return context
