from argparse import ArgumentParser
from pathlib import Path


def run_experiment(filename: str) -> bool: ...


def main() -> int:
    parser = ArgumentParser(
        description="Run a detection experiment from a YAML configuration."
    )
    parser.add_argument("config", type=Path, help="Experiment YAML configuration name")

    args = parser.parse_args()

    success = run_experiment(
        args.config,
    )
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
