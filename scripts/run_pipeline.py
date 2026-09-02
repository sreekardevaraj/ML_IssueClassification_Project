"""Run or watch the complete dataset-to-artifact ML pipeline."""

import argparse
import hashlib
import shutil
import subprocess
import sys
import time
from uuid import uuid4
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as data_file:
        for chunk in iter(lambda: data_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_pipeline(data_path: Path, artifact_dir: Path, tracking_uri: str, experiment: str, min_top1: float) -> None:
    staging_dir = artifact_dir.parent / f".{artifact_dir.name}.staging-{uuid4().hex}"
    lock_path = artifact_dir.parent / f".{artifact_dir.name}.lock"
    try:
        lock_path.mkdir()
    except FileExistsError:
        raise RuntimeError(f"Another pipeline is already running: {lock_path}")
    commands = [
        [sys.executable, str(ROOT / "scripts" / "train.py"), "--data", str(data_path), "--output-dir", str(staging_dir), "--tracking-uri", tracking_uri, "--experiment", experiment],
        [sys.executable, str(ROOT / "scripts" / "evaluate.py"), "--data", str(data_path), "--artifact-dir", str(staging_dir), "--tracking-uri", tracking_uri, "--experiment", experiment, "--min-top1", str(min_top1)],
    ]
    try:
        for command in commands:
            subprocess.run(command, cwd=ROOT, check=True)
        backup_dir = artifact_dir.parent / f".{artifact_dir.name}.previous"
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        if artifact_dir.exists():
            artifact_dir.rename(backup_dir)
        staging_dir.rename(artifact_dir)
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
    except Exception:
        if not artifact_dir.exists() and (artifact_dir.parent / f".{artifact_dir.name}.previous").exists():
            (artifact_dir.parent / f".{artifact_dir.name}.previous").rename(artifact_dir)
        raise
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        if lock_path.exists():
            lock_path.rmdir()


def wait_for_stable_file(path: Path, seconds: float = 1.0) -> str:
    first = fingerprint(path)
    time.sleep(seconds)
    second = fingerprint(path)
    if first != second:
        raise RuntimeError("Dataset is still being written; waiting for the next watch cycle")
    return second


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate when the dataset changes")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, default=Path("models/production"))
    parser.add_argument("--watch", action="store_true", help="Retrain whenever the dataset file changes")
    parser.add_argument("--interval", type=int, default=30, help="Watch interval in seconds")
    parser.add_argument("--tracking-uri", default="file:./mlruns")
    parser.add_argument("--experiment", default="support-case-classification")
    parser.add_argument("--min-top1", type=float, default=0.80)
    args = parser.parse_args()
    data_path = args.data if args.data.is_absolute() else ROOT / args.data
    artifact_dir = args.artifact_dir if args.artifact_dir.is_absolute() else ROOT / args.artifact_dir
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    if args.interval < 1:
        raise ValueError("--interval must be at least 1 second")

    previous = None
    while True:
        current = wait_for_stable_file(data_path)
        if current != previous:
            print(f"Dataset changed; running pipeline for {data_path}", flush=True)
            try:
                run_pipeline(data_path, artifact_dir, args.tracking_uri, args.experiment, args.min_top1)
                previous = current
            except Exception as error:
                if not args.watch:
                    raise
                print(f"Pipeline failed; will retry: {error}", file=sys.stderr, flush=True)
        if not args.watch:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
