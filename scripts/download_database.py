import shutil

import sys

from argparse import ArgumentParser

from pathlib import Path

import wfdb

from ecg_waveform.config import LUDB_DIR, MITDB_DIR

_MARKER_NAME = ".download_complete"

DATABASES: dict[str, Path] = {
    "mitdb": MITDB_DIR,
    "ludb": LUDB_DIR,
}


def _is_complete(output_dir: Path) -> bool:
    return (output_dir / _MARKER_NAME).exists()


def download_database(name: str, output_dir: Path, force: bool = False) -> bool:
    if force and output_dir.exists():
        print(f"Removing existing {name} directory at {output_dir}...")
        shutil.rmtree(output_dir)

    if _is_complete(output_dir):
        print(f"{name} already downloaded.")
        return True

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        print(f"Downloading {name}...")
        wfdb.dl_database(name, dl_dir=str(output_dir))
        (output_dir / _MARKER_NAME).touch()
        print(f"Done: {output_dir.resolve()}")
        return True
    except Exception as exc:
        shutil.rmtree(output_dir, ignore_errors=True)
        print(f"Failed to download {name}: {exc}", file=sys.stderr)
        return False


def main() -> int:
    parser = ArgumentParser(description="Download ECG databases from PhysioNet.")
    parser.add_argument(
        "database",
        choices=DATABASES.keys(),
        help="Database to download.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete and re-download the database even if it already exists.",
    )

    args = parser.parse_args()

    success = download_database(
        args.database,
        DATABASES[args.database],
        force=args.force,
    )

    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
