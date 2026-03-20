"""Recipe definitions for legal pipeline stage orderings.

A recipe declares which stages run and in what order. Optional stages
are skipped unless their activation condition is met (config flag or
manifest data).
"""

from typing import Dict, List, Optional, Set, TYPE_CHECKING

from ..utils.manifest import has_timing

if TYPE_CHECKING:
    import pandas as pd

RECIPES: Dict[str, List[str]] = {
    "pose": [
        "acquire",
        "manifest",
        "detect_person",
        "window_video",
        "clip_video",
        "crop_video",
        "extract",
        "normalize",
        "webdataset",
    ],
    "video": [
        "acquire",
        "manifest",
        "detect_person",
        "window_video",
        "clip_video",
        "crop_video",
        "obfuscate",
        "webdataset",
    ],
}

# Stages that are skipped unless explicitly enabled in config or manifest data
OPTIONAL_STAGES: Set[str] = {
    "detect_person",
    "clip_video",
    "crop_video",
    "obfuscate",
    "window_video",
}

# Static prerequisites for validation (not used for checkpoint hashing)
STAGE_PREREQUISITES: Dict[str, Set[str]] = {
    "crop_video": {"detect_person", "clip_video"},
    "normalize": {"extract"},
}


def get_steps(
    recipe: str,
    start_from: Optional[str] = None,
    stop_at: Optional[str] = None,
) -> List[str]:
    """Return the ordered stage list for a recipe, sliced by start/stop.

    Parameters
    ----------
    recipe : str
        Recipe name (must be a key in ``RECIPES``).
    start_from : str, optional
        Stage name to start from (inclusive).
    stop_at : str, optional
        Stage name to stop at (inclusive).

    Returns
    -------
    list of str
        Ordered stage names.

    Raises
    ------
    ValueError
        If *recipe* is unknown or *start_from*/*stop_at* not in the recipe.
    """
    if recipe not in RECIPES:
        raise ValueError(
            f"Unknown recipe '{recipe}'. Available: {sorted(RECIPES.keys())}"
        )

    steps = list(RECIPES[recipe])

    if start_from:
        if start_from not in steps:
            raise ValueError(
                f"--from='{start_from}' not found in recipe '{recipe}'. "
                f"Available stages: {steps}"
            )
        idx = steps.index(start_from)
        steps = steps[idx:]

    if stop_at:
        if stop_at not in steps:
            raise ValueError(
                f"--to='{stop_at}' not found in recipe '{recipe}'. "
                f"Available stages: {steps}"
            )
        idx = steps.index(stop_at)
        steps = steps[: idx + 1]

    return steps


def should_run_stage(
    stage_name: str,
    config: "object",
    manifest_df: Optional["pd.DataFrame"] = None,
) -> bool:
    """Determine whether an optional stage should execute.

    Non-optional stages always return True. Optional stages are activated
    by one of three mechanisms:

    1. **Explicit ``enabled`` flag** — ``detect_person``, ``crop_video``
    2. **Data-driven** — ``clip_video`` runs when manifest has START/END
    3. **Dict presence** — ``obfuscate``, ``window_video`` run when their
       key is present in ``config.stage_config``

    Parameters
    ----------
    stage_name : str
        The stage to check.
    config
        The top-level ``Config`` object.
    manifest_df : DataFrame, optional
        The current manifest (used for data-driven checks).

    Returns
    -------
    bool
        True if the stage should run.
    """
    if stage_name not in OPTIONAL_STAGES:
        return True

    if stage_name == "detect_person":
        return getattr(getattr(config, "detect_person", None), "enabled", False)

    if stage_name == "clip_video":
        # Data-driven: run if manifest has at least one row with both START and END
        if manifest_df is not None:
            return has_timing(manifest_df)
        # If no manifest yet (before manifest stage), assume it will have timing
        return True

    if stage_name == "crop_video":
        enabled = getattr(getattr(config, "crop_video", None), "enabled", False)
        if not enabled:
            return False
        # Also requires detect_person to have produced BBOX columns
        if manifest_df is not None:
            return "BBOX_X1" in manifest_df.columns
        return True

    if stage_name in ("obfuscate", "window_video"):
        stage_config = getattr(config, "stage_config", {})
        return stage_name in stage_config

    return False
