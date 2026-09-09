from __future__ import annotations

import logging
import os
import string
from typing import Any, Iterable, Optional

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.cluster import AgglomerativeClustering
import numpy as np

logger = logging.getLogger(__name__)

# NLTK resources
_REQUIRED_NLTK_RESOURCES = (
    ("tokenizers/punkt", "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("corpora/stopwords", "stopwords"),
)

def _ensure_nltk_resources() -> None:
    for resource_path, resource_name in _REQUIRED_NLTK_RESOURCES:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            logger.info("Downloading NLTK resource '%s'", resource_name)
            nltk.download(resource_name, quiet=True)

class TextClassifierError(Exception):
    pass

class TextClassifier:
    FILE_TYPE = "text"

    def __init__(
        self,
        model_path: Optional[str] = None,
        vector_size: int = 50,
        language: str = "english",
    ) -> None:
        _ensure_nltk_resources()
        self.vector_size = vector_size
        self._stemmer = PorterStemmer()
        self._stopwords = set(stopwords.words(language))
        self._punctuation = set(string.punctuation)
        self._model: Optional[Doc2Vec] = None

        if model_path and os.path.isfile(model_path):
            try:
                self._model = Doc2Vec.load(model_path)
            except Exception as exc:
                logger.warning("Could not load Doc2Vec model: %s", exc)

    def preprocess_text(self, raw_text: str) -> list[str]:
        if not isinstance(raw_text, str):
            raise TypeError("raw_text must be a string")
        try:
            tokens = word_tokenize(raw_text.lower())
        except LookupError as exc:
            raise TextClassifierError(f"NLTK tokenizer data missing: {exc}") from exc

        cleaned = []
        for token in tokens:
            token = token.strip(string.punctuation)
            if not token or token in self._stopwords or all(ch in self._punctuation for ch in token) or not any(ch.isalpha() for ch in token):
                continue
            cleaned.append(self._stemmer.stem(token))
        return cleaned

    def _init_bootstrap_model(self, seed_tokens: Iterable[str]) -> Doc2Vec:
        tagged = [TaggedDocument(words=list(seed_tokens), tags=["seed-0"])]
        model = Doc2Vec(vector_size=self.vector_size, min_count=1, epochs=20, workers=1)
        model.build_vocab(tagged)
        model.train(tagged, total_examples=model.corpus_count, epochs=model.epochs)
        return model

    def vectorize_text(self, processed_tokens: list[str]) -> list[float]:
        if not isinstance(processed_tokens, list):
            raise TypeError("processed_tokens must be a list of strings")
        if self._model is None:
            self._model = self._init_bootstrap_model(processed_tokens or ["doca"])
        try:
            vector = self._model.infer_vector(processed_tokens)
        except Exception as exc:
            raise TextClassifierError(f"Failed to infer vector: {exc}") from exc
        return [float(x) for x in vector.tolist()]

    def cluster_documents(self, documents_tokens: list[list[str]], n_clusters: int = 2) -> list[int]:
        """Apply Agglomerative Hierarchical clustering to categorize unlabeled text documents."""
        if not documents_tokens:
            return []
        vectors = [self.vectorize_text(tokens) for tokens in documents_tokens]
        # If there are fewer documents than requested clusters, adjust n_clusters
        actual_clusters = min(n_clusters, len(vectors))
        if actual_clusters < 1:
            return []

        # We need at least 2 samples for agglomerative clustering if n_clusters >= 2.
        # If we only have 1 document, we just return cluster 0.
        if len(vectors) == 1:
            return [0]

        clustering = AgglomerativeClustering(n_clusters=actual_clusters)
        labels = clustering.fit_predict(vectors)
        return labels.tolist()

    def process_file(self, file_path: str, mime_type: Optional[str] = None) -> dict[str, Any]:
        """Process a text document, returning a metadata dict for CouchDB."""
        if not file_path:
            raise ValueError("file_path must be a non-empty string")

        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
                raw_text = fh.read()
        except OSError as exc:
            raise TextClassifierError(f"Could not read {file_path}: {exc}") from exc

        tokens = self.preprocess_text(raw_text)
        vector = self.vectorize_text(tokens)

        try:
            file_size = os.path.getsize(file_path)
        except OSError:
            file_size = None

        return {
            "path": os.path.abspath(file_path),
            "file_name": os.path.basename(file_path),
            "file_type": self.FILE_TYPE,
            "extension": os.path.splitext(file_path)[1].lower(),
            "size_bytes": file_size,
            "token_count": len(tokens),
            "tokens": tokens,
            "vector": vector,
            "vector_size": len(vector),
        }
