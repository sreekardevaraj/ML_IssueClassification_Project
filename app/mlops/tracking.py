import hashlib
import json
from pathlib import Path
from typing import Any


def dataset_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as data_file:
        for chunk in iter(lambda: data_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def log_run(
    *,
    tracking_uri: str,
    experiment: str,
    run_name: str,
    parameters: dict[str, Any],
    metrics: dict[str, float],
    artifacts: list[Path],
    tags: dict[str, str],
) -> str:
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({key: str(value) for key, value in parameters.items()})
        mlflow.log_metrics(metrics)
        mlflow.set_tags(tags)
        for artifact in artifacts:
            if artifact.exists():
                mlflow.log_artifact(str(artifact), artifact_path="model_bundle")
        return run.info.run_id


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def log_monitoring_event(
    *,
    tracking_uri: str,
    experiment: str,
    run_name: str,
    metrics: dict[str, float],
    parameters: dict[str, Any] | None = None,
    tags: dict[str, str] | None = None,
) -> str | None:
    """Best-effort MLflow logging for operational monitoring.

    Inference should not fail because a local MLflow server is unavailable, so
    monitoring events are intentionally non-blocking from the caller's point of view.
    """
    try:
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)
        with mlflow.start_run(run_name=run_name) as run:
            if parameters:
                mlflow.log_params({key: str(value) for key, value in parameters.items()})
            mlflow.log_metrics(metrics)
            if tags:
                mlflow.set_tags(tags)
            return run.info.run_id
    except Exception:
        return None
