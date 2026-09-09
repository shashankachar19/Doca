"""Image file handler for the DoCA project.

Provides an :class:`ImageHandler` that reads an image file, extracts any
embedded text with Tesseract OCR, computes SIFT keypoints with OpenCV,
and returns a JSON-safe metadata dictionary suitable for
:meth:`handlers.db_handler.DBHandler.save_document`.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import cv2
import numpy as np
import pytesseract

logger = logging.getLogger(__name__)

# Windows default install location for Tesseract OCR. Only applied when
# the binary actually exists at that path so the module still imports
# cleanly on Linux/macOS or on a machine where Tesseract lives elsewhere.
_WINDOWS_TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(_WINDOWS_TESSERACT_CMD):
    pytesseract.pytesseract.tesseract_cmd = _WINDOWS_TESSERACT_CMD
    logger.debug("Using Tesseract binary at %s", _WINDOWS_TESSERACT_CMD)

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


class ImageHandlerError(Exception):
    """Raised when the ImageHandler cannot complete an operation."""


class ImageHandler:
    """Extract text and visual features from image files.

    Parameters
    ----------
    tesseract_cmd:
        Optional absolute path to the ``tesseract`` binary. Useful on
        Windows or when the binary is not on ``PATH``.
    ocr_lang:
        Tesseract language code(s), e.g. ``"eng"`` or ``"eng+fra"``.
    """

    FILE_TYPE = "image"

    def __init__(
        self,
        tesseract_cmd: Optional[str] = None,
        ocr_lang: str = "eng",
    ) -> None:
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        self.ocr_lang = ocr_lang

        try:
            self._sift = cv2.SIFT_create()
        except AttributeError as exc:
            # SIFT moved between opencv-contrib versions; surface a clear error.
            raise ImageHandlerError(
                "cv2.SIFT_create() is unavailable. Install a recent "
                f"opencv-python (>=4.4): {exc}"
            ) from exc

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _read_grayscale(self, image_path: str) -> np.ndarray:
        """Load ``image_path`` in grayscale or raise ImageHandlerError."""
        if not os.path.isfile(image_path):
            raise ImageHandlerError(f"Image file not found: {image_path}")

        try:
            # cv2.imread returns None for unreadable/corrupt files instead
            # of raising, so we need an explicit check.
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        except cv2.error as exc:
            raise ImageHandlerError(
                f"OpenCV failed to read '{image_path}': {exc}"
            ) from exc

        if image is None:
            raise ImageHandlerError(
                f"OpenCV could not decode image '{image_path}' "
                "(unsupported format or corrupted file)."
            )
        return image

    # ------------------------------------------------------------------ #
    # OCR
    # ------------------------------------------------------------------ #
    def extract_text(self, image_path: str) -> str:
        """Return OCR text for ``image_path`` with whitespace collapsed."""
        image = self._read_grayscale(image_path)

        try:
            raw = pytesseract.image_to_string(image, lang=self.ocr_lang)
        except pytesseract.TesseractNotFoundError as exc:
            raise ImageHandlerError(
                "Tesseract binary not found. Install tesseract-ocr or set "
                f"ImageHandler(tesseract_cmd=...): {exc}"
            ) from exc
        except pytesseract.TesseractError as exc:
            logger.warning("Tesseract failed on %s: %s", image_path, exc)
            return ""
        except RuntimeError as exc:
            logger.warning("OCR runtime error on %s: %s", image_path, exc)
            return ""

        # Collapse runs of whitespace/newlines and trim.
        return " ".join(raw.split()).strip()

    # ------------------------------------------------------------------ #
    # Feature extraction
    # ------------------------------------------------------------------ #
    def extract_features(self, image_path: str) -> int:
        """Return the number of SIFT keypoints detected in the image.

        The raw descriptor matrix is not persisted. Descriptors for a
        single image can easily exceed several megabytes, which bloats
        CouchDB documents beyond its recommended size.
        """
        image = self._read_grayscale(image_path)

        try:
            keypoints, _descriptors = self._sift.detectAndCompute(image, None)
        except cv2.error as exc:
            logger.warning("SIFT failed on %s: %s", image_path, exc)
            return 0

        return len(keypoints) if keypoints is not None else 0

    # ------------------------------------------------------------------ #
    # Orchestration
    # ------------------------------------------------------------------ #
    def process_file(
        self,
        file_path: str,
        mime_type: Optional[str] = None,
    ) -> dict[str, Any]:
        """Run OCR + SIFT on ``file_path`` and return JSON-safe metadata.

        ``mime_type`` is accepted for a uniform content-based call signature
        across all handlers; it is recorded in the result but not used to
        alter OCR/feature extraction.
        """
        if not file_path:
            raise ValueError("file_path must be a non-empty string")
        _ = mime_type  # reserved for future format-specific behavior

        extension = os.path.splitext(file_path)[1].lower()
        if extension and extension not in SUPPORTED_EXTENSIONS:
            logger.debug("Unusual image extension '%s'; attempting anyway.", extension)

        # Fail fast if the image can't be decoded. Both sub-steps would
        # raise the same error; doing it once yields a cleaner trace.
        self._read_grayscale(file_path)

        extracted_text = self.extract_text(file_path)
        keypoint_count = self.extract_features(file_path)

        try:
            size_bytes = os.path.getsize(file_path)
        except OSError:
            size_bytes = None

        return {
            "path": os.path.abspath(file_path),
            "file_name": os.path.basename(file_path),
            "file_type": self.FILE_TYPE,
            "extension": extension,
            "size_bytes": size_bytes,
            "ocr_text": extracted_text,
            "ocr_text_length": len(extracted_text),
            "sift_keypoint_count": int(keypoint_count),
        }
