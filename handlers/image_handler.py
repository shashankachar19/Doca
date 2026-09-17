"""Image file handler for the DoCA project.

Provides an :class:`ImageHandler` that reads an image file, extracts any
embedded text with Tesseract OCR, computes SIFT keypoints with OpenCV,
and returns a JSON-safe metadata dictionary suitable for
:meth:`handlers.db_handler.DBHandler.save_document`.

OCR preprocessing pipeline
--------------------------
Dark-mode screenshots (light text on dark background) defeat naive OCR
because Tesseract expects dark-on-light text.  The handler now runs a
multi-pass OCR strategy:

1. **Original grayscale** — standard OCR pass.
2. **Inverted + adaptive threshold** — detects dark backgrounds (mean
   intensity < 127), inverts, applies adaptive Gaussian thresholding and
   morphological cleanup to produce clean dark-on-white text.

The pass yielding the most extracted text wins.  A ``has_text`` flag is
set when OCR produces ≥ 3 words, enabling downstream classifiers to
route these images to a Text/Document category.
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

# Minimum word count to set the has_text flag.
# Lowered to 1 so sparse dark-mode UI screenshots still qualify.
MIN_WORD_COUNT_FOR_TEXT = 1


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
    min_word_count:
        Minimum number of words OCR must extract for ``has_text`` to be
        ``True``.  Lowered to 3 (from an implicit ~10) so that sparse
        dark-mode screenshots still qualify.
    """

    FILE_TYPE = "image"

    def __init__(
        self,
        tesseract_cmd: Optional[str] = None,
        ocr_lang: str = "eng",
        min_word_count: int = MIN_WORD_COUNT_FOR_TEXT,
    ) -> None:
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        self.ocr_lang = ocr_lang
        self.min_word_count = min_word_count

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
    # OCR preprocessing for dark-mode / low-contrast images
    # ------------------------------------------------------------------ #
    @staticmethod
    def _is_dark_background(gray: np.ndarray, threshold: int = 127) -> bool:
        """Return True if the image's mean intensity suggests a dark background."""
        return float(np.mean(gray)) < threshold

    @staticmethod
    def _preprocess_for_ocr(gray: np.ndarray) -> np.ndarray:
        """Produce a clean dark-on-white binary image for Tesseract.

        Steps:
        1. If the background is dark (mean < 127), invert the image so
           that text becomes dark on a light background.
        2. Apply adaptive Gaussian thresholding to handle uneven lighting
           and gradient backgrounds common in dark-mode UIs.
        3. Apply a small morphological close to join broken character
           strokes caused by anti-aliasing or thin fonts.
        """
        work = gray.copy()

        # Step 1: Invert dark images
        if ImageHandler._is_dark_background(work):
            work = cv2.bitwise_not(work)

        # Step 2: Adaptive threshold — handles gradient backgrounds
        work = cv2.adaptiveThreshold(
            work,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=15,   # neighbourhood size (must be odd)
            C=8,            # constant subtracted from mean
        )

        # Step 3: Morphological close to repair thin/broken strokes
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        work = cv2.morphologyEx(work, cv2.MORPH_CLOSE, kernel)

        return work

    def _ocr_single(self, image: np.ndarray) -> str:
        """Run Tesseract on a single prepared image and return cleaned text."""
        try:
            raw = pytesseract.image_to_string(image, lang=self.ocr_lang)
        except pytesseract.TesseractNotFoundError as exc:
            raise ImageHandlerError(
                "Tesseract binary not found. Install tesseract-ocr or set "
                f"ImageHandler(tesseract_cmd=...): {exc}"
            ) from exc
        except pytesseract.TesseractError as exc:
            logger.warning("Tesseract failed: %s", exc)
            return ""
        except RuntimeError as exc:
            logger.warning("OCR runtime error: %s", exc)
            return ""
        return " ".join(raw.split()).strip()

    # ------------------------------------------------------------------ #
    # OCR — multi-pass strategy
    # ------------------------------------------------------------------ #
    def extract_text(self, image_path: str) -> str:
        """Return OCR text for ``image_path``.

        Runs two passes:
        1. Original grayscale image.
        2. Preprocessed (inverted + adaptive-threshold + morph-close).

        Returns whichever pass extracted more text, giving dark-mode
        screenshots a fair shot.
        """
        gray = self._read_grayscale(image_path)

        # Pass 1: original
        text_original = self._ocr_single(gray)

        # Pass 2: preprocessed for dark/low-contrast backgrounds
        try:
            preprocessed = self._preprocess_for_ocr(gray)
            text_preprocessed = self._ocr_single(preprocessed)
        except cv2.error as exc:
            logger.debug("Preprocessing failed for %s: %s", image_path, exc)
            text_preprocessed = ""

        # Keep whichever pass extracted more text.
        if len(text_preprocessed) > len(text_original):
            logger.debug(
                "Preprocessed OCR yielded more text for %s "
                "(%d vs %d chars)",
                image_path, len(text_preprocessed), len(text_original),
            )
            return text_preprocessed
        return text_original

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
        word_count = len(extracted_text.split()) if extracted_text else 0
        has_text = word_count >= self.min_word_count

        logger.info(
            "OCR result for '%s': word_count=%d, has_text=%s, text=%r",
            os.path.basename(file_path), word_count, has_text,
            (extracted_text[:200] + '...') if len(extracted_text) > 200 else extracted_text,
        )

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
            "ocr_word_count": word_count,
            "has_text": has_text,
            "sift_keypoint_count": int(keypoint_count),
        }
