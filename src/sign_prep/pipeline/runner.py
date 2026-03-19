"""Recipe-driven pipeline runner."""

import logging
from pathlib import Path
from typing import Optional, Set

from ..config.schema import Config
from ..registry import DATASET_REGISTRY, PROCESSOR_REGISTRY
from ..utils.manifest import read_manifest
from .checkpoint import (
    check_success,
    compute_manifest_hash,
    compute_stage_hash,
    compute_upstream_hash,
    success_content_hash,
    write_success,
)
from .context import PipelineContext
from .recipes import get_steps, should_run_stage, OPTIONAL_STAGES, RECIPES

logger = logging.getLogger(__name__)

# Stages that produce a new manifest (context.manifest_path changes after them)
_MANIFEST_PRODUCING_STAGES = {"manifest", "detect_person", "window_video"}


class PipelineRunner:
    """Execute a recipe-driven pipeline with context routing.

    The runner resolves the stage list from the recipe, delegates
    ``acquire`` and ``manifest`` to the dataset adapter, and dispatches
    remaining stages to registered processors.

    After each stage the runner updates ``context.manifest_path``,
    ``context.video_dir``, and the corresponding ``*_producer`` fields
    so that downstream stages read from the correct artifacts.
    """

    def __init__(
        self,
        config: Config,
        force_stage: Optional[str] = None,
        force_all: bool = False,
    ):
        self.config = config
        self.dataset = DATASET_REGISTRY[config.dataset]()
        self.force_all = force_all
        # Build set of stages to force (named stage + all downstream)
        self._forced_stages: Set[str] = set()
        if force_stage:
            # Validate that the force stage exists in the recipe
            recipe_stages = RECIPES.get(config.recipe, [])
            if recipe_stages and force_stage not in recipe_stages:
                raise ValueError(
                    f"--force '{force_stage}' is not a stage in recipe "
                    f"'{config.recipe}'. Available: {recipe_stages}"
                )
            all_steps = list(
                get_steps(config.recipe, start_from=force_stage)
            )
            self._forced_stages = set(all_steps)

    def run(self) -> PipelineContext:
        steps = get_steps(
            self.config.recipe,
            self.config.start_from,
            self.config.stop_at,
        )

        context = PipelineContext(
            config=self.config,
            dataset=self.dataset,
            project_root=Path(self.config.paths.root).parent.parent,
            # Initialize routing from config paths
            manifest_path=Path(self.config.paths.manifest) if self.config.paths.manifest else None,
            video_dir=Path(self.config.paths.videos) if self.config.paths.videos else None,
        )

        # Seed routing for --from/--only: replay upstream stages so context
        # points to the correct artifact locations.
        self._seed_routing(context, steps)

        logger.info("=" * 70)
        logger.info(
            "Pipeline: dataset=%s  recipe=%s  steps=%s",
            self.config.dataset,
            self.config.recipe,
            steps,
        )
        logger.info("=" * 70)

        for step_name in steps:
            # Check optional stage activation
            if not should_run_stage(step_name, self.config, context.manifest_df):
                logger.info("Skipping optional stage: %s", step_name)
                continue

            # Resolve output directory for checkpoint and stage_output_dir
            output_dir = self._get_stage_output_dir(step_name)

            # Checkpoint: skip if already completed (unless forced)
            if not self._is_forced(step_name):
                if output_dir and self._check_checkpoint(step_name, output_dir, context):
                    logger.info("Checkpoint valid, skipping: %s", step_name)
                    self._update_routing(step_name, context)
                    self._reload_manifest_if_needed(step_name, context)
                    context.completed_steps.append(step_name)
                    continue

            logger.info("Running step: %s", step_name)

            # Set stage_output_dir so processors know where to write
            context.stage_output_dir = output_dir

            if step_name == "acquire":
                context = self.dataset.acquire(self.config, context)
            elif step_name == "manifest":
                context = self.dataset.build_manifest(self.config, context)
            else:
                if step_name not in PROCESSOR_REGISTRY:
                    if step_name in OPTIONAL_STAGES:
                        logger.warning(
                            "Skipping stage '%s': no processor registered. "
                            "Install the required package or register a processor.",
                            step_name,
                        )
                        continue
                    raise ValueError(
                        f"Unknown processor '{step_name}'. "
                        f"Available: {list(PROCESSOR_REGISTRY.keys())}"
                    )
                processor = PROCESSOR_REGISTRY[step_name](self.config)

                # Validate inputs before running
                processor.validate_inputs(context)

                if not processor.validate(context):
                    raise RuntimeError(f"Validation failed for {step_name}")

                context = processor.run(context)

            # Update context routing after stage completes
            self._update_routing(step_name, context)
            context.completed_steps.append(step_name)

            # Write checkpoint marker
            if output_dir:
                self._write_checkpoint(step_name, output_dir, context)

            step_stats = context.stats.get(step_name, {})
            logger.info("Completed: %s | Stats: %s", step_name, step_stats)

        logger.info("Pipeline complete.")
        return context

    def _is_forced(self, step_name: str) -> bool:
        """Return True if this stage should be forced to re-run."""
        return self.force_all or step_name in self._forced_stages

    def _seed_routing(self, context: PipelineContext, steps: list) -> None:
        """Replay routing for upstream stages when using --from/--only.

        When the pipeline starts at a stage other than the first in the
        recipe, we need to set context routing fields as if the earlier
        stages had already run, so that the first active stage reads from
        the correct artifact locations.
        """
        full_steps = RECIPES[self.config.recipe]
        first_step = steps[0]
        if first_step == full_steps[0]:
            return  # Starting from the beginning, no seeding needed

        first_idx = full_steps.index(first_step)
        for stage in full_steps[:first_idx]:
            if should_run_stage(stage, self.config, context.manifest_df):
                self._update_routing(stage, context)
                # Reload manifest so subsequent routing checks have data
                self._reload_manifest_if_needed(stage, context)

    def _get_stage_output_dir(self, step_name: str) -> Optional[Path]:
        """Return the output directory for a stage's checkpoint marker."""
        cfg = self.config
        root = Path(cfg.paths.root)

        # Stages with dedicated checkpoint directories under root
        if step_name == "acquire":
            return root / "acquire" / cfg.run_name

        if step_name == "manifest":
            return root / "manifest" / cfg.run_name

        if step_name == "detect_person":
            return root / "detect_person" / cfg.run_name

        if step_name == "window_video":
            return root / "window_video" / cfg.run_name

        if step_name == "obfuscate":
            return root / "obfuscated" / cfg.run_name

        # Stages whose output dir is a config path attribute
        _config_path_attrs = {
            "clip_video": "clips",
            "crop_video": "cropped_clips",
            "extract": "landmarks",
            "normalize": "normalized",
            "webdataset": "webdataset",
        }
        config_attr = _config_path_attrs.get(step_name)
        if config_attr:
            path_val = getattr(cfg.paths, config_attr, None)
            if path_val:
                return Path(path_val)

        return None

    def _check_checkpoint(
        self,
        step_name: str,
        output_dir: Path,
        context: PipelineContext,
    ) -> bool:
        """Check if a stage's checkpoint is still valid."""
        stage_hash = compute_stage_hash(self.config, step_name)
        manifest_hash = self._current_manifest_hash(context)
        upstream_hash = self._current_upstream_hash(context)
        return check_success(output_dir, stage_hash, manifest_hash, upstream_hash)

    def _write_checkpoint(
        self,
        step_name: str,
        output_dir: Path,
        context: PipelineContext,
    ) -> None:
        """Write a checkpoint marker after a stage completes."""
        stage_hash = compute_stage_hash(self.config, step_name)
        manifest_hash = self._current_manifest_hash(context)
        upstream_hash = self._current_upstream_hash(context)

        step_stats = context.stats.get(step_name, {})
        output_count = step_stats.get("total", step_stats.get("written", 1))

        write_success(
            output_dir=output_dir,
            stage_name=step_name,
            stage_config_hash=stage_hash,
            manifest_hash=manifest_hash,
            upstream_hash=upstream_hash,
            output_count=max(output_count, 1),
        )

    def _current_manifest_hash(self, context: PipelineContext) -> str:
        """Compute manifest hash from the current context."""
        if context.manifest_df is not None:
            return compute_manifest_hash(context.manifest_df)
        if context.manifest_path and context.manifest_path.exists():
            return compute_manifest_hash(context.manifest_path)
        return "sha256:none"

    def _current_upstream_hash(self, context: PipelineContext) -> str:
        """Compute upstream hash from completed stages' success markers."""
        if not context.completed_steps:
            return ""
        hashes = []
        for prev_step in context.completed_steps:
            prev_dir = self._get_stage_output_dir(prev_step)
            if prev_dir:
                hashes.append(success_content_hash(prev_dir))
        if not hashes:
            return ""
        return compute_upstream_hash(hashes)

    def _reload_manifest_if_needed(
        self, step_name: str, context: PipelineContext,
    ) -> None:
        """Reload manifest_df from disk when skipping a manifest-producing stage.

        When a checkpoint-valid stage is skipped, the runner still updates
        routing (manifest_path changes), but manifest_df would be stale.
        This reloads it so downstream stages and activation checks see the
        correct data.
        """
        if step_name not in _MANIFEST_PRODUCING_STAGES:
            return
        if context.manifest_path and context.manifest_path.exists():
            context.manifest_df = read_manifest(
                str(context.manifest_path), normalize_columns=True,
            )

    def _update_routing(self, step_name: str, context: PipelineContext) -> None:
        """Update context routing fields after a stage completes.

        This keeps ``manifest_path`` / ``video_dir`` and their producer
        fields in lockstep so downstream stages read from the correct
        artifacts.
        """
        cfg = self.config
        run_name = cfg.run_name

        if step_name == "manifest":
            context.manifest_path = Path(cfg.paths.manifest)
            context.video_dir = Path(cfg.paths.videos)
            context.manifest_producer = "manifest"
            context.video_dir_producer = "manifest"

        elif step_name == "detect_person":
            root = Path(cfg.paths.root)
            stage_manifest = root / "detect_person" / run_name / "manifest.csv"
            context.manifest_path = stage_manifest
            context.manifest_producer = "detect_person"

        elif step_name == "window_video":
            root = Path(cfg.paths.root)
            stage_manifest = root / "window_video" / run_name / "manifest.csv"
            context.manifest_path = stage_manifest
            context.manifest_producer = "window_video"

        elif step_name == "clip_video":
            context.video_dir = Path(cfg.paths.clips)
            context.video_dir_producer = "clip_video"

        elif step_name == "crop_video":
            context.video_dir = Path(cfg.paths.cropped_clips)
            context.video_dir_producer = "crop_video"

        elif step_name == "obfuscate":
            root = Path(cfg.paths.root)
            context.video_dir = root / "obfuscated" / run_name
            context.video_dir_producer = "obfuscate"
