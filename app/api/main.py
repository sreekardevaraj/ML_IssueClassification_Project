from contextlib import asynccontextmanager
import json
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.settings import get_settings
from app.domain.schemas import BatchRequest, BatchResponse, CaseInput, ModelInfo, Prediction
from app.domain.taxonomy import TAXONOMY_VERSION
from app.inference.service import ClassificationService
from app.models.adapters import TfidfEmbedder, XGBoostClassifier

settings = get_settings()


def build_service() -> ClassificationService:
    if settings.artifact_dir:
        artifact_dir = Path(settings.artifact_dir)
        labels_path = artifact_dir / "labels.json"
        embedding_config_path = artifact_dir / "embedding_config.json"
        tfidf_path = artifact_dir / "tfidf.joblib"
        xgboost_path = artifact_dir / "xgboost.json"
        if not all(path.exists() for path in (labels_path, embedding_config_path, tfidf_path, xgboost_path)):
            return ClassificationService(settings=settings, embedder=TfidfEmbedder())
        labels = json.loads(labels_path.read_text(encoding="utf-8"))
        embedding_config = json.loads(embedding_config_path.read_text(encoding="utf-8"))
        return ClassificationService(
            settings=settings,
            embedder=TfidfEmbedder(vectorizer_path=str(tfidf_path)),
            classifier=XGBoostClassifier(model_path=str(xgboost_path), labels=labels),
        )
    return ClassificationService(settings=settings, embedder=TfidfEmbedder())


service = build_service()


def persist_prediction(prediction: Prediction, case_text: str | None = None) -> None:
    results_dir = Path(settings.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    record = prediction.model_dump()
    if settings.persist_case_text and case_text is not None:
        record["case_text"] = case_text
    with (results_dir / "predictions.jsonl").open("a", encoding="utf-8") as results_file:
        results_file.write(json.dumps(record) + "\n")


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield


app = FastAPI(title=settings.app_name, version=settings.model_version, lifespan=lifespan)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request.state.request_id = request.headers.get("X-Request-ID", str(uuid4()))
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    return response


@app.exception_handler(ValueError)
async def value_error_handler(_: Request, exc: ValueError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(RuntimeError)
async def runtime_error_handler(_: Request, exc: RuntimeError):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.get("/health/live")
def live() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health/ready")
def ready() -> dict[str, bool | str]:
    return {"status": "ready" if service.artifacts_loaded else "not_ready", "artifacts_loaded": service.artifacts_loaded}


@app.post("/v1/predict", response_model=Prediction)
def predict(request: Request, case: CaseInput) -> Prediction:
    prediction = service.predict(case, request_id=request.state.request_id)
    persist_prediction(prediction, case.text)
    return prediction


@app.post("/v1/predict/batch", response_model=BatchResponse)
def predict_batch(request: Request, batch: BatchRequest) -> BatchResponse:
    predictions = [service.predict(case, request_id=request.state.request_id) for case in batch.cases]
    for prediction, case in zip(predictions, batch.cases):
        persist_prediction(prediction, case.text)
    return BatchResponse(predictions=predictions)


@app.get("/v1/model-info", response_model=ModelInfo)
def model_info() -> ModelInfo:
    return ModelInfo(
        taxonomy_version=TAXONOMY_VERSION,
        model_version=settings.model_version,
        first_stage="TF-IDF features + XGBoost",
        ambiguity_path="DeBERTa embeddings (optional ambiguity layer)",
        artifacts_loaded=service.artifacts_loaded,
    )
