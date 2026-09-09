"""Text file handler for the DoCA project.

Provides a :class:`TextHandler` that reads a plain-text document,
preprocesses it with NLTK (tokenize, lowercase, strip punctuation,
remove stop words, stem) and converts the resulting tokens into a
Doc2Vec vector using Gensim.

The output is a plain-Python dict so it can be stored directly in
CouchDB via :class:`handlers.db_handler.DBHandler`.
"""

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

logger = logging.getLogger(__name__)

# NLTK resources that preprocess_text depends on.
_REQUIRED_NLTK_RESOURCES = (
    ("tokenizers/punkt", "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("corpora/stopwords", "stopwords"),
)

# Extensions handled by dedicated Office extractors rather than plain-text read.
_DOCX_EXTENSIONS = {".docx"}
_XLSX_EXTENSIONS = {".xlsx", ".xlsm"}
_PPTX_EXTENSIONS = {".pptx", ".pptm"}

# Office MIME types we recognize. Legacy binary formats (.doc/.xls/.ppt) are
# acknowledged but not extracted here; they require textract/antiword/etc.
_OFFICE_MIME_EXT = {
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
}


class TextHandlerError(Exception):
    """Raised when the TextHandler cannot complete an operation."""


def _ensure_nltk_resources() -> None:
    """Download the NLTK corpora we rely on if they are missing."""
    for resource_path, resource_name in _REQUIRED_NLTK_RESOURCES:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            logger.info("Downloading NLTK resource '%s'", resource_name)
            nltk.download(resource_name, quiet=True)


class TextHandler:
    """Classify and vectorize plain-text documents.

    Parameters
    ----------
    model_path:
        Optional path to a pre-trained Doc2Vec model. If provided and the
        file exists, the model is loaded on instantiation. Otherwise, a
        small bootstrap model is initialized on first use.
    vector_size:
        Dimensionality of the Doc2Vec embeddings.
    language:
        Stop-word language passed to NLTK. Defaults to English.
    """

    FILE_TYPE = "text"

    def __init__(
        self,
        model_path: Optional[str] = None,
        vector_size: int = 50,
        language: str = "english",
    ) -> None:
        _ensure_nltk_resources()

        self.model_path = model_path
        self.vector_size = vector_size
        self._stemmer = PorterStemmer()
        self._stopwords = set(stopwords.words(language))
        self._punctuation = set(string.punctuation)
        self._model: Optional[Doc2Vec] = None

        if model_path and os.path.isfile(model_path):
            try:
                self._model = Doc2Vec.load(model_path)
                logger.info("Loaded Doc2Vec model from %s", model_path)
            except Exception as exc:  # gensim raises many types here
                logger.warning(
                    "Could not load Doc2Vec model at %s: %s", model_path, exc
                )
                self._model = None

    # ------------------------------------------------------------------ #
    # Preprocessing
    # ------------------------------------------------------------------ #
    def preprocess_text(self, raw_text: str) -> list[str]:
        """Tokenize, lowercase, strip punctuation and stop words, then stem."""
        if not isinstance(raw_text, str):
            raise TypeError("raw_text must be a string")

        try:
            tokens = word_tokenize(raw_text.lower())
        except LookupError as exc:
            raise TextHandlerError(
                f"NLTK tokenizer data is missing: {exc}"
            ) from exc

        cleaned: list[str] = []
        for token in tokens:
            # Strip any leading/trailing punctuation characters.
            token = token.strip(string.punctuation)
            if not token:
                continue
            if token in self._stopwords:
                continue
            if all(ch in self._punctuation for ch in token):
                continue
            if not any(ch.isalpha() for ch in token):
                continue
            cleaned.append(self._stemmer.stem(token))
        return cleaned


    # ------------------------------------------------------------------ #
    # Vectorization
    # ------------------------------------------------------------------ #
    def _init_bootstrap_model(self, seed_tokens: Iterable[str]) -> Doc2Vec:
        """Create a minimal Doc2Vec model seeded on the first document.

        This exists so the handler is usable out-of-the-box. For real
        workloads, train on the full corpus and pass ``model_path``.
        """
        tagged = [TaggedDocument(words=list(seed_tokens), tags=["seed-0"])]
        try:
            model = Doc2Vec(
                vector_size=self.vector_size,
                min_count=1,
                epochs=20,
                workers=1,
            )
            model.build_vocab(tagged)
            model.train(
                tagged,
                total_examples=model.corpus_count,
                epochs=model.epochs,
            )
        except Exception as exc:
            raise TextHandlerError(
                f"Failed to initialize Doc2Vec model: {exc}"
            ) from exc
        logger.info("Initialized bootstrap Doc2Vec model (dim=%d)", self.vector_size)
        return model

    def vectorize_text(self, processed_tokens: list[str]) -> list[float]:
        """Return a Doc2Vec embedding for the given tokens as a plain list."""
        if not isinstance(processed_tokens, list):
            raise TypeError("processed_tokens must be a list of strings")

        if self._model is None:
            self._model = self._init_bootstrap_model(processed_tokens or ["doca"])

        try:
            vector = self._model.infer_vector(processed_tokens)
        except Exception as exc:
            raise TextHandlerError(
                f"Failed to infer document vector: {exc}"
            ) from exc

        # Convert numpy floats to native Python floats so CouchDB can
        # serialize the payload as JSON.
        return [float(x) for x in vector.tolist()]


    # ------------------------------------------------------------------ #
    # Orchestration
    # ------------------------------------------------------------------ #
    def _read_text_file(self, file_path: str) -> str:
        """Read a text file, tolerating non-UTF-8 bytes."""
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
                return fh.read()
        except FileNotFoundError as exc:
            raise TextHandlerError(f"File not found: {file_path}") from exc
        except OSError as exc:
            raise TextHandlerError(
                f"Could not read file '{file_path}': {exc}"
            ) from exc

    # ------------------------------------------------------------------ #
    # Office document extractors
    # ------------------------------------------------------------------ #
    @staticmethod
    def _extract_docx(file_path: str) -> str:
        """Return the concatenated text of paragraphs and tables in a .docx."""
        try:
            from docx import Document  # python-docx
        except ImportError as exc:
            raise TextHandlerError(
                "python-docx is required to read .docx files. "
                "Install it with 'pip install python-docx'."
            ) from exc

        try:
            doc = Document(file_path)
        except Exception as exc:
            raise TextHandlerError(
                f"Could not open .docx '{file_path}': {exc}"
            ) from exc

        chunks: list[str] = [p.text for p in doc.paragraphs if p.text]
        # Pull any text sitting inside tables too.
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    if cell.text:
                        chunks.append(cell.text)
        return "\n".join(chunks)

    @staticmethod
    def _extract_xlsx(file_path: str) -> str:
        """Return the string values of every cell across every sheet."""
        try:
            from openpyxl import load_workbook
        except ImportError as exc:
            raise TextHandlerError(
                "openpyxl is required to read .xlsx files. "
                "Install it with 'pip install openpyxl'."
            ) from exc

        try:
            # data_only=True so formulas are replaced with their last cached value.
            workbook = load_workbook(file_path, data_only=True, read_only=True)
        except Exception as exc:
            raise TextHandlerError(
                f"Could not open .xlsx '{file_path}': {exc}"
            ) from exc

        chunks: list[str] = []
        try:
            for sheet in workbook.worksheets:
                chunks.append(f"# Sheet: {sheet.title}")
                for row in sheet.iter_rows(values_only=True):
                    row_strs = [str(v) for v in row if v is not None]
                    if row_strs:
                        chunks.append("\t".join(row_strs))
        finally:
            workbook.close()
        return "\n".join(chunks)

    @staticmethod
    def _extract_pptx(file_path: str) -> str:
        """Return the text of every shape frame across every slide."""
        try:
            from pptx import Presentation  # python-pptx
        except ImportError as exc:
            raise TextHandlerError(
                "python-pptx is required to read .pptx files. "
                "Install it with 'pip install python-pptx'."
            ) from exc

        try:
            prs = Presentation(file_path)
        except Exception as exc:
            raise TextHandlerError(
                f"Could not open .pptx '{file_path}': {exc}"
            ) from exc

        chunks: list[str] = []
        for slide_idx, slide in enumerate(prs.slides, start=1):
            chunks.append(f"# Slide {slide_idx}")
            for shape in slide.shapes:
                if not getattr(shape, "has_text_frame", False):
                    continue
                for paragraph in shape.text_frame.paragraphs:
                    text = "".join(run.text for run in paragraph.runs)
                    if text:
                        chunks.append(text)
            # Speaker notes are often full of searchable content.
            notes = getattr(slide, "notes_slide", None)
            if notes is not None and notes.notes_text_frame is not None:
                note_text = notes.notes_text_frame.text
                if note_text:
                    chunks.append(f"[notes] {note_text}")
        return "\n".join(chunks)

    def _extract_text(self, file_path: str, mime_type: Optional[str]) -> str:
        """Pick the right extractor for ``file_path`` and return its raw text.

        Extension match wins over MIME so a misreported content type can't
        send a .docx through the plain-text path, but MIME is consulted as
        a fallback for extensionless uploads.
        """
        ext = os.path.splitext(file_path)[1].lower()
        if not ext and mime_type:
            ext = _OFFICE_MIME_EXT.get(mime_type.lower().strip(), ext)

        if ext in _DOCX_EXTENSIONS:
            logger.debug("Extracting text from .docx: %s", file_path)
            return self._extract_docx(file_path)
        if ext in _XLSX_EXTENSIONS:
            logger.debug("Extracting text from .xlsx: %s", file_path)
            return self._extract_xlsx(file_path)
        if ext in _PPTX_EXTENSIONS:
            logger.debug("Extracting text from .pptx: %s", file_path)
            return self._extract_pptx(file_path)
        return self._read_text_file(file_path)

    def process_file(
        self,
        file_path: str,
        mime_type: Optional[str] = None,
    ) -> dict[str, Any]:
        """Run the full pipeline on ``file_path`` and return a metadata dict.

        Parameters
        ----------
        file_path:
            Path to the document to process.
        mime_type:
            Optional MIME type hint (from libmagic). Used only when the
            file has no extension. Callers that already detect MIME, like
            :class:`DocumentSorter`, can pass it through here.

        Returns
        -------
        dict
            A JSON-serializable metadata dictionary suitable for
            :meth:`handlers.db_handler.DBHandler.save_document`.
        """
        if not file_path:
            raise ValueError("file_path must be a non-empty string")

        raw_text = self._extract_text(file_path, mime_type)
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
