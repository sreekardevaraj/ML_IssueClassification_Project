"""Evaluate a trained TF-IDF + XGBoost bundle on a labeled CSV."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from app.domain.taxonomy import CATEGORY_BY_NAME
from app.mlops.tracking import dataset_fingerprint, log_run
from app.models.adapters import TfidfEmbedder, XGBoostClassifier


def top_k_accuracy(probabilities, expected, k: int) -> float:
    return float((probabilities.argsort(axis=1)[:, -k:] == expected[:, None]).any(axis=1).mean())


def evaluate(data_path: Path, artifact_dir: Path, tracking_uri: str, experiment: str, min_top1: float) -> dict:
    data = pd.read_csv(data_path).dropna(subset=["text", "label"]).drop_duplicates()
    labels = json.loads((artifact_dir / "labels.json").read_text(encoding="utf-8"))
    label_to_id = {label: index for index, label in enumerate(labels)}
    data = data[data["label"].isin(CATEGORY_BY_NAME)]
    _, test = train_test_split(data, test_size=0.2, random_state=42, stratify=data["label"])
    expected = test["label"].map(label_to_id).to_numpy()
    config = json.loads((artifact_dir / "embedding_config.json").read_text(encoding="utf-8"))
    embedder = TfidfEmbedder(vectorizer_path=str(artifact_dir / "tfidf.joblib"))
    classifier = XGBoostClassifier(model_path=str(artifact_dir / "xgboost.json"), labels=labels)
    probabilities = classifier.predict_proba(embedder.embed(test["text"].tolist()))
    predicted = probabilities.argmax(axis=1)
    report = {
        "samples": len(test),
        "top_1_accuracy": top_k_accuracy(probabilities, expected, 1),
        "top_2_accuracy": top_k_accuracy(probabilities, expected, 2),
        "top_3_accuracy": top_k_accuracy(probabilities, expected, 3),
        "macro_f1": float(f1_score(expected, predicted, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(expected, predicted, average="weighted", zero_division=0)),
        "classification_report": classification_report(
            expected, predicted, labels=list(range(len(labels))), target_names=labels,
            output_dict=True, zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(expected, predicted, labels=list(range(len(labels)))).tolist(),
        "dataset_sha256": dataset_fingerprint(data_path),
    }
    (artifact_dir / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    run_id = log_run(
        tracking_uri=tracking_uri,
        experiment=experiment,
        run_name="evaluate",
        parameters={"samples": len(test), "min_top1_accuracy": min_top1},
        metrics={key: report[key] for key in ("top_1_accuracy", "top_2_accuracy", "top_3_accuracy", "macro_f1", "weighted_f1")},
        artifacts=[artifact_dir / "evaluation.json", artifact_dir / "confusion_matrix.json"],
        tags={"stage": "evaluation", "dataset_sha256": report["dataset_sha256"]},
    )
    report["mlflow_run_id"] = run_id
    (artifact_dir / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if report["top_1_accuracy"] < min_top1:
        raise ValueError(f"Quality gate failed: Top-1 {report['top_1_accuracy']:.4f} < {min_top1:.4f}")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--tracking-uri", default="file:./mlruns")
    parser.add_argument("--experiment", default="support-case-classification")
    parser.add_argument("--min-top1", type=float, default=0.80)
    args = parser.parse_args()
    evaluate(args.data, args.artifact_dir, args.tracking_uri, args.experiment, args.min_top1)
