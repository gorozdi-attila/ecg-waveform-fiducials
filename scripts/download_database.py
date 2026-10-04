import shutil
import sys
from argparse import ArgumentParser
from pathlib import Path

import wfdb

from ecg_waveform.config import DATA_DIR

_MARKER_NAME = ".download_complete"


def download_database(
    database_name: str, output_dir: Path, force: bool = False
) -> bool:
    output_dir = output_dir / database_name

    if force and output_dir.exists():
        print(f"Removing existing {database_name} directory at {output_dir}...")
        shutil.rmtree(output_dir)

    if (output_dir / _MARKER_NAME).exists():
        print(f"{database_name} already downloaded.")
        return True

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        print(f"Downloading {database_name}...")
        wfdb.dl_database(database_name, dl_dir=str(output_dir))
        (output_dir / _MARKER_NAME).touch()
        print(f"Done: {output_dir.resolve()}")
        return True

    except Exception as e:
        shutil.rmtree(output_dir, ignore_errors=True)
        print(f"Failed to download {database_name}: {e}", file=sys.stderr)
        return False


def main() -> int:
    parser = ArgumentParser(description="Download ECG databases from PhysioNet.")
    parser.add_argument(
        "database_name",
        type=str,
        help="Database to download.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete and re-download the database even if it already exists.",
    )

    args = parser.parse_args()

    success = download_database(
        args.database_name,
        DATA_DIR,
        force=args.force,
    )

    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
