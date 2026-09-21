"""Compare label and text-length distributions for a candidate dataset."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.mlops.tracking import log_monitoring_event


def report(
    reference_path: Path,
    candidate_path: Path,
    output_path: Path,
    label_threshold: float = 0.05,
    text_length_threshold: float = 0.20,
) -> dict:
    reference = pd.read_csv(reference_path)
    candidate = pd.read_csv(candidate_path)
    for frame, name in ((reference, "reference"), (candidate, "candidate")):
        if not {"text", "label"}.issubset(frame.columns):
            raise ValueError(f"{name} dataset must contain text and label columns")
    reference_counts = reference["label"].value_counts(normalize=True)
    candidate_counts = candidate["label"].value_counts(normalize=True)
    labels = sorted(set(reference_counts.index) | set(candidate_counts.index))
    label_deltas = {
        label: abs(float(reference_counts.get(label, 0)) - float(candidate_counts.get(label, 0)))
        for label in labels
    }
    reference_text_length = float(reference["text"].astype(str).str.len().mean())
    candidate_text_length = float(candidate["text"].astype(str).str.len().mean())
    text_length_delta_ratio = (
        abs(candidate_text_length - reference_text_length) / reference_text_length
        if reference_text_length
        else 0.0
    )
    max_label_delta = max(label_deltas.values(), default=0.0)
    drift_alert = max_label_delta > label_threshold or text_length_delta_ratio > text_length_threshold
    result = {
        "reference_records": len(reference),
        "candidate_records": len(candidate),
        "label_distribution": {
            label: {"reference": float(reference_counts.get(label, 0)), "candidate": float(candidate_counts.get(label, 0))}
            for label in labels
        },
        "mean_text_length": {
            "reference": reference_text_length,
            "candidate": candidate_text_length,
        },
        "max_label_distribution_delta": max_label_delta,
        "text_length_delta_ratio": text_length_delta_ratio,
        "label_threshold": label_threshold,
        "text_length_threshold": text_length_threshold,
        "drift_alert": drift_alert,
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
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--experiment", default="support-case-monitoring")
    parser.add_argument("--label-threshold", type=float, default=0.05)
    parser.add_argument("--text-length-threshold", type=float, default=0.20)
    args = parser.parse_args()
    result = report(
        args.reference,
        args.candidate,
        args.output,
        args.label_threshold,
        args.text_length_threshold,
    )
    log_monitoring_event(
        tracking_uri=args.tracking_uri,
        experiment=args.experiment,
        run_name="drift_report",
        metrics={
            "max_label_distribution_delta": result["max_label_distribution_delta"],
            "text_length_delta_ratio": result["text_length_delta_ratio"],
            "drift_alert_count": 1.0 if result["drift_alert"] else 0.0,
        },
        parameters={
            "reference": args.reference,
            "candidate": args.candidate,
            "label_threshold": args.label_threshold,
            "text_length_threshold": args.text_length_threshold,
        },
        tags={"stage": "drift_monitoring"},
    )
