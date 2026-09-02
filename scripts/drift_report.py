"""Compare label and text-length distributions for a candidate dataset."""

import argparse
import json
from pathlib import Path

import pandas as pd


def report(reference_path: Path, candidate_path: Path, output_path: Path) -> dict:
    reference = pd.read_csv(reference_path)
    candidate = pd.read_csv(candidate_path)
    for frame, name in ((reference, "reference"), (candidate, "candidate")):
        if not {"text", "label"}.issubset(frame.columns):
            raise ValueError(f"{name} dataset must contain text and label columns")
    reference_counts = reference["label"].value_counts(normalize=True)
    candidate_counts = candidate["label"].value_counts(normalize=True)
    labels = sorted(set(reference_counts.index) | set(candidate_counts.index))
    result = {
        "reference_records": len(reference),
        "candidate_records": len(candidate),
        "label_distribution": {
            label: {"reference": float(reference_counts.get(label, 0)), "candidate": float(candidate_counts.get(label, 0))}
            for label in labels
        },
        "mean_text_length": {
            "reference": float(reference["text"].astype(str).str.len().mean()),
            "candidate": float(candidate["text"].astype(str).str.len().mean()),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/drift.json"))
    args = parser.parse_args()
    report(args.reference, args.candidate, args.output)