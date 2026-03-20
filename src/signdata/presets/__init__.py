"""Named keypoint reduction presets.

Provides immutable, named bundles of keypoint indices for common
extractor-to-reduced-set mappings.  Eliminates duplicated index lists
across YAML config files (~400 lines of YAML replaced by preset names).
"""

from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Preset definitions
# ---------------------------------------------------------------------------

KEYPOINT_PRESETS: Dict[str, dict] = {
    "mediapipe_553_to_85": {
        "description": (
            "MediaPipe refined: 6 pose + 37 face + 21 left hand + 21 right hand"
        ),
        "source_keypoints": 553,
        "target_keypoints": 85,
        "indices": (
            # Pose (offset=0): shoulders, elbows, hips
            [11, 12, 13, 14, 23, 24]
            # Face (offset=33): 37 selected landmarks (with iris refinement)
            + [
                33, 37, 46, 47, 50, 66, 70, 72, 79, 85, 88, 94, 97,
                114, 115, 126, 166, 184, 185, 192, 205, 211, 214,
                296, 302, 309, 315, 318, 324, 327, 344, 356,
                395, 419, 430, 501, 506,
            ]
            # Left hand (offset=511): all 21
            + list(range(511, 532))
            # Right hand (offset=532): all 21
            + list(range(532, 553))
        ),
    },
    "mediapipe_543_to_83": {
        "description": (
            "MediaPipe unrefined: 6 pose + 35 face + 21 left hand + 21 right hand"
        ),
        "source_keypoints": 543,
        "target_keypoints": 83,
        "indices": (
            # Pose (offset=0): shoulders, elbows, hips
            [11, 12, 13, 14, 23, 24]
            # Face (offset=33): 35 selected landmarks (no iris)
            + [
                33, 37, 46, 47, 50, 66, 70, 72, 79, 85, 88, 94, 97,
                114, 115, 126, 166, 184, 185, 192, 205, 211, 214,
                296, 302, 309, 315, 318, 324, 327, 344, 356,
                395, 419, 430,
            ]
            # Left hand (offset=501): all 21
            + list(range(501, 522))
            # Right hand (offset=522): all 21
            + list(range(522, 543))
        ),
    },
    "mmpose_133_to_85": {
        "description": (
            "COCO WholeBody 133 to 85 selected upper body, face, and hand keypoints"
        ),
        "source_keypoints": 133,
        "target_keypoints": 85,
        "indices": (
            # Upper body pose
            [5, 6, 7, 8, 11, 12]
            # Face (selected)
            + [23, 25, 27, 29, 31, 33, 35, 37, 39]
            + [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
            + [52, 54, 56, 58]
            # Hands (selected + remaining)
            + [71, 73, 75, 77, 79, 81, 83, 84, 85, 86, 87, 88, 89, 90]
            + list(range(91, 133))
        ),
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def resolve_keypoint_indices(
    preset_name: Optional[str] = None,
    manual_indices: Optional[List[int]] = None,
) -> Optional[List[int]]:
    """Resolve keypoint indices from a preset name or manual list.

    Priority order:
    1. *preset_name* — look up in ``KEYPOINT_PRESETS``
    2. *manual_indices* — return as-is
    3. Neither — return ``None`` (no reduction)

    Parameters
    ----------
    preset_name : str, optional
        Name of a registered keypoint preset.
    manual_indices : list of int, optional
        Explicit list of keypoint indices.

    Returns
    -------
    list of int or None
        The resolved keypoint indices, or ``None`` if no reduction.

    Raises
    ------
    ValueError
        If *preset_name* is not a recognized preset.
    """
    if preset_name is not None:
        if preset_name not in KEYPOINT_PRESETS:
            raise ValueError(
                f"Unknown keypoint preset '{preset_name}'. "
                f"Available: {sorted(KEYPOINT_PRESETS.keys())}"
            )
        return list(KEYPOINT_PRESETS[preset_name]["indices"])

    if manual_indices is not None:
        return list(manual_indices)

    return None


def list_presets() -> Dict[str, str]:
    """Return a mapping of preset names to their descriptions.

    Useful for CLI discoverability (e.g. ``--list-presets``).
    """
    return {
        name: info["description"]
        for name, info in KEYPOINT_PRESETS.items()
    }
