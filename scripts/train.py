"""Train the TF-IDF + XGBoost first-stage classifier.

Example:
  python scripts/train.py --data data/cases.csv --output-dir models/production
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

from app.domain.taxonomy import CATEGORY_BY_NAME
from app.mlops.tracking import dataset_fingerprint, log_run
from app.models.adapters import TfidfEmbedder


def train(data_path: Path, output_dir: Path, seed: int, tracking_uri: str, experiment: str) -> None:
    data = pd.read_csv(data_path)
    required = {"text", "label"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    data = data[["text", "label"]].dropna().drop_duplicates()
    data["text"] = data["text"].astype(str).str.strip()
    data = data[data["text"].str.len() >= 3]
    unknown = sorted(set(data["label"]) - set(CATEGORY_BY_NAME))
    if unknown:
        raise ValueError(f"Unknown labels: {unknown}")
    if data["label"].nunique() != len(CATEGORY_BY_NAME):
        raise ValueError("Training data must contain all 17 taxonomy categories")

    labels = sorted(CATEGORY_BY_NAME)
    label_to_id = {label: index for index, label in enumerate(labels)}
    y = data["label"].map(label_to_id).to_numpy()
    train_text, test_text, train_y, test_y = train_test_split(
        data["text"].to_numpy(), y, test_size=0.2, random_state=seed, stratify=y
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    embedder = TfidfEmbedder()
    embedder.fit(train_text.tolist())
    features = embedder.embed(train_text.tolist())
    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=len(labels),
        n_estimators=150,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=2.0,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=seed,
        n_jobs=4,
    )
    model.fit(features, train_y)

    model.get_booster().save_model(output_dir / "xgboost.json")
    embedder.save(output_dir / "tfidf.joblib")
    (output_dir / "embedding_config.json").write_text(
        json.dumps({"backend": embedder.backend, "dimension": embedder.dimension}, indent=2),
        encoding="utf-8",
    )
    (output_dir / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
    (output_dir / "manifest.json").write_text(
        json.dumps({
            "model_version": "xgboost-tfidf-1",
            "embedding_backend": embedder.backend,
            "labels": labels,
            "seed": seed,
            "records_total": len(data),
            "records_train": len(train_text),
            "records_test": len(test_text),
            "dataset_sha256": dataset_fingerprint(data_path),
        }, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame({"text": train_text, "label": [labels[index] for index in train_y]}).to_csv(output_dir / "train_records.csv", index=False)
    pd.DataFrame({"text": test_text, "label": [labels[index] for index in test_y]}).to_csv(output_dir / "test_records.csv", index=False)
    run_id = log_run(
        tracking_uri=tracking_uri,
        experiment=experiment,
        run_name="train",
        parameters={"seed": seed, "records_total": len(data), "features": embedder.dimension},
        metrics={},
        artifacts=[output_dir / "xgboost.json", output_dir / "tfidf.joblib", output_dir / "manifest.json"],
        tags={"stage": "training", "dataset_sha256": dataset_fingerprint(data_path)},
    )
    (output_dir / "manifest.json").write_text(
        json.dumps({**json.loads((output_dir / "manifest.json").read_text()), "mlflow_run_id": run_id}, indent=2),
        encoding="utf-8",
    )
    print(f"Saved artifacts to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        type=Path,
        required=True
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/production")
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--tracking-uri",
        default="sqlite:///mlflow.db"
    )

    parser.add_argument(
        "--experiment",
        default="support-case-classification"
    )

    args = parser.parse_args()

    train(
        args.data,
        args.output_dir,
        args.seed,
        args.tracking_uri,
        args.experiment
    )
