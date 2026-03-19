"""Entry point: python -m sign_prep run|experiment <config.yaml>"""

import logging
import sys

# Ensure registrations happen on import
import sign_prep.datasets  # noqa: F401
import sign_prep.processors  # noqa: F401
import sign_prep.extractors  # noqa: F401

from sign_prep.cli import parse_args
from sign_prep.config import load_config
from sign_prep.pipeline import PipelineRunner


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    args = parse_args()

    if args.command is None:
        print("Usage: python -m sign_prep <command> <config.yaml> [options]")
        print()
        print("Commands:")
        print("  run         Run a single preprocessing job")
        print("  experiment  Run a multi-job experiment")
        print()
        print("Run 'python -m sign_prep <command> --help' for details.")
        sys.exit(1)

    if args.command == "run":
        # Handle --list-presets early exit (no config file needed)
        if args.list_presets:
            from sign_prep.presets import list_presets
            for name, desc in sorted(list_presets().items()):
                print(f"  {name:30s} {desc}")
            return

        if not args.config:
            print("Error: config file is required (unless using --list-presets)")
            sys.exit(1)

        overrides = args.override or []

        # Wire CLI flags into config overrides
        if args.start_from:
            overrides.append(f"start_from={args.start_from}")
        if args.stop_at:
            overrides.append(f"stop_at={args.stop_at}")
        if args.only:
            overrides.append(f"start_from={args.only}")
            overrides.append(f"stop_at={args.only}")
        if args.run_name:
            overrides.append(f"run_name={args.run_name}")

        config = load_config(args.config, overrides=overrides or None)

        # PipelineRunner validates --force against recipe stages
        runner = PipelineRunner(
            config,
            force_stage=args.force,
            force_all=args.force_all,
        )
        runner.run()

    elif args.command == "experiment":
        from sign_prep.config.experiment import load_experiment
        from sign_prep.pipeline.experiment import ExperimentRunner

        experiment = load_experiment(args.config)
        runner = ExperimentRunner(
            experiment, force_all=args.force_all,
        )
        results = runner.run()

        # Exit with error code if any job failed
        failed = sum(1 for r in results if r.status == "failed")
        if failed:
            sys.exit(1)


if __name__ == "__main__":
    main()
