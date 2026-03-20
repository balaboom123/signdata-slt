"""Experiment runner — execute multiple pipeline jobs sequentially."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config.experiment import ExperimentConfig, _flatten_overrides
from ..config.loader import load_config
from .runner import PipelineRunner

logger = logging.getLogger(__name__)


@dataclass
class JobResult:
    """Result of a single job within an experiment."""

    config: str
    status: str  # "success" or "failed"
    stats: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class ExperimentRunner:
    """Execute an ordered list of pipeline jobs.

    Each job is loaded from its own config YAML with optional overrides
    applied.  Jobs run sequentially; if a job fails, the error is logged
    and subsequent jobs still execute.  A summary is printed at the end.

    Args:
        experiment: Validated experiment config.
        force_all: If True, pass ``force_all=True`` to every PipelineRunner.
    """

    def __init__(
        self,
        experiment: ExperimentConfig,
        force_all: bool = False,
    ):
        self.experiment = experiment
        self.force_all = force_all

    def run(self) -> List[JobResult]:
        """Run all jobs and return per-job results."""
        n_jobs = len(self.experiment.jobs)

        logger.info("=" * 70)
        logger.info("Experiment: %s", self.experiment.name)
        if self.experiment.description:
            logger.info("  %s", self.experiment.description)
        logger.info("Jobs: %d", n_jobs)
        logger.info("=" * 70)

        results: List[JobResult] = []

        for idx, job in enumerate(self.experiment.jobs, start=1):
            job_label = Path(job.config).name
            logger.info(
                "[%d/%d] Running: %s", idx, n_jobs, job_label,
            )

            # Flatten nested overrides to dot-separated keys
            dict_overrides = _flatten_overrides(job.overrides) if job.overrides else None

            if dict_overrides:
                override_strs = [f"{k}={v}" for k, v in dict_overrides.items()]
                logger.info("  Overrides: %s", ", ".join(override_strs))

            try:
                config = load_config(
                    job.config, dict_overrides=dict_overrides,
                )
                runner = PipelineRunner(
                    config, force_all=self.force_all,
                )
                context = runner.run()

                results.append(JobResult(
                    config=job.config,
                    status="success",
                    stats=context.stats,
                ))
                logger.info("[%d/%d] Completed: %s", idx, n_jobs, job_label)

            except Exception as e:
                logger.error(
                    "[%d/%d] Failed: %s — %s", idx, n_jobs, job_label, e,
                )
                results.append(JobResult(
                    config=job.config,
                    status="failed",
                    error=str(e),
                ))

        # Summary
        succeeded = sum(1 for r in results if r.status == "success")
        failed = sum(1 for r in results if r.status == "failed")

        logger.info("=" * 70)
        logger.info(
            "Experiment complete: %d/%d succeeded, %d failed",
            succeeded, n_jobs, failed,
        )

        if failed:
            failed_jobs = [
                Path(r.config).name for r in results if r.status == "failed"
            ]
            logger.error("Failed jobs: %s", ", ".join(failed_jobs))

        logger.info("=" * 70)

        return results
