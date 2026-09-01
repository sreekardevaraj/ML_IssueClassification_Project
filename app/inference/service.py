from dataclasses import dataclass
from uuid import uuid4

from app.core.settings import Settings
from app.domain.schemas import Candidate, CaseInput, Prediction
from app.domain.taxonomy import CATEGORIES, TAXONOMY_VERSION
from app.models.adapters import TfidfEmbedder, XGBoostClassifier


@dataclass
class ClassificationService:
    settings: Settings
    embedder: TfidfEmbedder
    classifier: XGBoostClassifier | None = None

    @property
    def artifacts_loaded(self) -> bool:
        return self.classifier is not None and self.classifier._model is not None and self.embedder.dimension > 0

    def predict(self, case: CaseInput, request_id: str | None = None) -> Prediction:
        if not self.artifacts_loaded:
            raise RuntimeError("Model artifacts are not ready; train and publish a TF-IDF/XGBoost bundle")
        ranked = self._rank_trained(case.text)
        confidence_semantics = "XGBoost class probability; calibrate on validation data"
        top, second = ranked[0], ranked[1]
        margin = top[1] - second[1]
        escalate = top[1] < self.settings.first_stage_threshold or margin < self.settings.margin_threshold
        reason = None
        stage = "tfidf_xgboost"
        if escalate:
            reason = "low_confidence" if top[1] < self.settings.first_stage_threshold else "small_top1_top2_margin"
            stage = "deberta" if self.settings.enable_llm else "abstained"
        return Prediction(
            case_id=case.case_id,
            category=top[0],
            candidates=[Candidate(category=name, score=score) for name, score in ranked[:3]],
            stage=stage,
            confidence=top[1],
            confidence_semantics=confidence_semantics,
            routing_reason=reason,
            taxonomy_version=TAXONOMY_VERSION,
            model_version=self.settings.model_version,
            request_id=request_id or str(uuid4()),
        )

    def _rank_trained(self, text: str) -> list[tuple[str, float]]:
        features = self.embedder.embed([text])
        probabilities = self.classifier.predict_proba(features)[0]
        labels = self.classifier.labels or [category.name for category in CATEGORIES]
        return sorted(zip(labels, probabilities.tolist()), key=lambda item: item[1], reverse=True)

