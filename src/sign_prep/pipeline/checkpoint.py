"""Stage completion markers and config-aware checkpoint validation.

Each pipeline stage writes a ``_SUCCESS.json`` marker to its output
directory on completion.  On subsequent runs the runner checks these
markers to skip stages whose inputs haven't changed.

Hashing strategy
----------------
* **stage_config_hash** — SHA-256 of the config fields that affect this
  stage's output (declared per-processor via ``config_hash_fields``).
* **manifest_hash** — SHA-256 of the manifest file content (or DataFrame
  bytes) so changes in the input data invalidate the checkpoint.
* **upstream_success_hash** — SHA-256 of the ``_SUCCESS.json`` hashes of
  the stages that actually produced the artifacts this stage consumed
  (context-producer-based, not a static table).
"""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import pandas as pd

# ---------------------------------------------------------------------------
# Filename constant
# ---------------------------------------------------------------------------

SUCCESS_FILENAME = "_SUCCESS.json"

# ---------------------------------------------------------------------------
# Per-stage hash field registry
#
# Each key is a stage name; the value lists the config dot-paths whose
# values are hashed to produce ``stage_config_hash``.
# ---------------------------------------------------------------------------

STAGE_HASH_FIELDS: Dict[str, List[str]] = {
    "acquire": [
        "source",
    ],
    "manifest": [
        "source",
    ],
    "detect_person": [
        "detect_person.model",
        "detect_person.backend",
        "detect_person.confidence_threshold",
        "detect_person.sample_strategy",
        "detect_person.uniform_frames",
        "detect_person.max_frames",
        "detect_person.device",
        "detect_person.min_bbox_area",
        "processing.frame_skip",
        "processing.signer_policy",
    ],
    "clip_video": [
        "clip_video.codec",
        "clip_video.resize",
        "processing.min_duration",
        "processing.max_duration",
    ],
    "crop_video": [
        "crop_video.padding",
        "crop_video.codec",
    ],
    "extract": [
        "extractor.name",
        # MediaPipe-specific
        "extractor.model_complexity",
        "extractor.min_detection_confidence",
        "extractor.min_tracking_confidence",
        "extractor.refine_face_landmarks",
        # MMPose-specific
        "extractor.pose_model_config",
        "extractor.pose_model_checkpoint",
        "extractor.det_model_config",
        "extractor.det_model_checkpoint",
        "extractor.bbox_threshold",
        "extractor.keypoint_threshold",
        "extractor.add_visible",
        # Shared
        "extractor.batch_size",
        "extractor.device",
        "processing.target_fps",
        "processing.frame_skip",
        "processing.accept_fps_range",
        "processing.min_duration",
        "processing.max_duration",
        "processing.signer_policy",
    ],
    "normalize": [
        "normalize.mode",
        "normalize.remove_z",
        "normalize.select_keypoints",
        "normalize.keypoint_preset",
        "normalize.keypoint_indices",
        "normalize.mask_empty_frames",
        "normalize.mask_low_confidence",
        "normalize.visibility_threshold",
        "normalize.missing_value",
    ],
    "webdataset": [
        "webdataset.max_shard_count",
        "webdataset.max_shard_size",
        "recipe",
    ],
}


# ---------------------------------------------------------------------------
# Hashing helpers
# ---------------------------------------------------------------------------

def _resolve_dotpath(obj: Any, dotpath: str) -> Any:
    """Resolve a dot-separated path on a nested object / dict.

    Examples::

        _resolve_dotpath(config, "extractor.name")  # config.extractor.name
        _resolve_dotpath({"a": {"b": 1}}, "a.b")    # 1
    """
    current = obj
    for part in dotpath.split("."):
        if isinstance(current, dict):
            current = current.get(part)
        else:
            current = getattr(current, part, None)
        if current is None:
            return None
    return current


def _stable_json(value: Any) -> str:
    """Return a deterministic JSON string for hashing."""
    return json.dumps(value, sort_keys=True, default=str)


def compute_stage_hash(
    config: Any,
    stage_name: str,
    hash_fields: Optional[List[str]] = None,
) -> str:
    """Compute a SHA-256 hash of the config fields relevant to *stage_name*.

    Parameters
    ----------
    config
        The top-level ``Config`` object (or any nested-attribute-accessible
        object / dict).
    stage_name : str
        Pipeline stage name (used to look up ``STAGE_HASH_FIELDS`` when
        *hash_fields* is not provided).
    hash_fields : list of str, optional
        Explicit list of config dot-paths to hash.  Overrides the registry.

    Returns
    -------
    str
        ``"sha256:<hex>"`` digest string.
    """
    fields = hash_fields or STAGE_HASH_FIELDS.get(stage_name, [])
    values = {f: _resolve_dotpath(config, f) for f in sorted(fields)}
    payload = _stable_json(values)
    digest = hashlib.sha256(payload.encode()).hexdigest()
    return f"sha256:{digest}"


def compute_manifest_hash(
    manifest: Union[str, Path, pd.DataFrame],
) -> str:
    """Compute a SHA-256 hash of manifest content.

    Accepts either a file path (hashes raw bytes) or a DataFrame (hashes
    the TSV serialization).

    Returns
    -------
    str
        ``"sha256:<hex>"`` digest string.
    """
    if isinstance(manifest, pd.DataFrame):
        payload = manifest.to_csv(sep="\t", index=False).encode()
    else:
        path = Path(manifest)
        if not path.exists():
            return "sha256:missing"
        payload = path.read_bytes()

    digest = hashlib.sha256(payload).hexdigest()
    return f"sha256:{digest}"


def success_content_hash(
    output_dir: Union[str, Path],
) -> str:
    """Compute a SHA-256 hash of an existing ``_SUCCESS.json`` file.

    This captures the *full* state of a completed stage (config hash,
    manifest hash, upstream hash, timestamp) — not just the config hash.
    Use the return value as input to ``compute_upstream_hash`` so that
    downstream stages detect *any* upstream re-execution, even when the
    upstream config itself didn't change.

    Returns ``"sha256:missing"`` if the marker does not exist.
    """
    path = Path(output_dir) / SUCCESS_FILENAME
    if not path.exists():
        return "sha256:missing"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return f"sha256:{digest}"


def compute_upstream_hash(
    upstream_hashes: Sequence[str],
) -> str:
    """Compute a combined hash from upstream stages' success markers.

    Parameters
    ----------
    upstream_hashes : sequence of str
        A hash for each upstream stage.  Callers should use
        ``success_content_hash(output_dir)`` to obtain these — that
        hashes the *entire* ``_SUCCESS.json`` content, so any upstream
        re-execution (even with the same config but a changed manifest or
        changed upstream inputs) produces a different value.

    Returns
    -------
    str
        ``"sha256:<hex>"`` digest string.
    """
    combined = "\n".join(sorted(upstream_hashes))
    digest = hashlib.sha256(combined.encode()).hexdigest()
    return f"sha256:{digest}"


# ---------------------------------------------------------------------------
# Write / read / check
# ---------------------------------------------------------------------------

def write_success(
    output_dir: Union[str, Path],
    stage_name: str,
    stage_config_hash: str,
    manifest_hash: str,
    upstream_hash: str = "",
    output_count: int = 1,
    output_sample: Optional[List[str]] = None,
) -> Path:
    """Write a ``_SUCCESS.json`` marker to *output_dir*.

    *output_count* defaults to 1 so that callers that omit it (e.g.
    single-file stages like ``manifest``) still produce a marker that
    ``check_success`` will accept (it rejects ``output_count <= 0``).

    Returns the path to the written file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    marker = {
        "stage": stage_name,
        "stage_config_hash": stage_config_hash,
        "manifest_hash": manifest_hash,
        "upstream_success_hash": upstream_hash,
        "output_count": output_count,
        "output_sample": output_sample or [],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    path = output_dir / SUCCESS_FILENAME
    path.write_text(json.dumps(marker, indent=2))
    return path


def read_success(
    output_dir: Union[str, Path],
) -> Optional[Dict[str, Any]]:
    """Read and parse a ``_SUCCESS.json`` marker.

    Returns *None* if the file does not exist or is malformed.
    """
    path = Path(output_dir) / SUCCESS_FILENAME
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def check_success(
    output_dir: Union[str, Path],
    stage_config_hash: str,
    manifest_hash: str,
    upstream_hash: str = "",
) -> bool:
    """Validate an existing ``_SUCCESS.json`` against current hashes.

    Returns *True* only when all of the following hold:

    1. ``_SUCCESS.json`` exists and is valid JSON.
    2. ``stage_config_hash`` matches.
    3. ``manifest_hash`` matches.
    4. ``upstream_success_hash`` matches (if provided).
    5. ``output_count > 0``.
    6. At least one file from ``output_sample`` still exists on disk.
    """
    marker = read_success(output_dir)
    if marker is None:
        return False

    if marker.get("stage_config_hash") != stage_config_hash:
        return False

    if marker.get("manifest_hash") != manifest_hash:
        return False

    if upstream_hash and marker.get("upstream_success_hash") != upstream_hash:
        return False

    if marker.get("output_count", 0) <= 0:
        return False

    # Spot-check: at least one sample file still exists
    samples = marker.get("output_sample", [])
    if samples:
        output_dir = Path(output_dir)
        if not any((output_dir / s).exists() for s in samples):
            return False

    return True
