from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


class TextEmbedder(Protocol):
    model_version: str

    def embed(self, texts: list[str]) -> np.ndarray: ...


@dataclass
class TfidfEmbedder:
    vectorizer_path: str | None = None
    max_features: int = 5_000

    def __post_init__(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import FeatureUnion

        self._vectorizer = None
        self._fitted = False
        if self.vectorizer_path:
            import joblib

            self._vectorizer = joblib.load(Path(self.vectorizer_path))
            self._fitted = True
        else:
            self._vectorizer = TfidfVectorizer(
                lowercase=True, strip_accents="unicode", analyzer="word", ngram_range=(1, 2),
                min_df=1, max_features=self.max_features, sublinear_tf=True,
            )
            self._vectorizer = FeatureUnion([
                ("word", self._vectorizer),
                ("char", TfidfVectorizer(
                    lowercase=True, analyzer="char_wb", ngram_range=(3, 5),
                    min_df=2, max_features=self.max_features, sublinear_tf=True,
                )),
            ])

    @property
    def backend(self) -> str:
        return "tfidf"

    @property
    def dimension(self) -> int:
        if not self._fitted:
            return 0
        return int(self._vectorizer.transform([""]).shape[1])

    def fit(self, texts: list[str]) -> None:
        self._vectorizer.fit(texts)
        self._fitted = True

    def save(self, path: Path) -> None:
        import joblib

        joblib.dump(self._vectorizer, path)

    def embed(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TF-IDF vectorizer is not fitted")
        return self._vectorizer.transform(texts).astype(np.float32)


@dataclass
class XGBoostClassifier:
    model_path: str | None = None
    labels: list[str] | None = None

    def __post_init__(self) -> None:
        self._model = None
        if self.model_path:
            import xgboost as xgb

            self._model = xgb.Booster()
            self._model.load_model(str(Path(self.model_path)))

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("XGBoost artifact is not configured")
        import xgboost as xgb

        probabilities = self._model.predict(xgb.DMatrix(features))
        return np.asarray(probabilities, dtype=np.float32)


@dataclass
class DebertaEmbedder:
    model_version: str = "deberta-unloaded"
    model_name: str = "microsoft/deberta-v3-base"
    max_length: int = 256

    def embed(self, texts: list[str]) -> np.ndarray:
        import torch
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModel.from_pretrained(self.model_name)
        encoded = tokenizer(texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        with torch.no_grad():
            output = model(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1)
        return (output * mask).sum(dim=1).div(mask.sum(dim=1).clamp(min=1)).cpu().numpy()


@dataclass
class LlamaDecisionAdapter:
    model_version: str = "llama-3.3-70b"

    def decide(self, text: str, candidates: list[str]) -> str | None:
        raise RuntimeError("Llama provider is not configured; set SUPPORT_LLAMA_ENDPOINT before enabling it")
