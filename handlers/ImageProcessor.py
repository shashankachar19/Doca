"""Image/PDF processor for the DoCA project (watchdog pipeline).

Provides an :class:`ImageProcessor` that handles both raster images and
PDFs.  For images it extracts text via Tesseract OCR with a multi-pass
preprocessing pipeline (see below), and computes SIFT keypoints.  For
PDFs it renders each page to a raster via PyMuPDF and runs the same
OCR pipeline.

OCR preprocessing pipeline
--------------------------
Dark-mode screenshots (light text on dark background) defeat naive OCR
because Tesseract expects dark-on-light text.  The processor now runs a
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
import fitz  # PyMuPDF
import numpy as np
import pytesseract

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
SUPPORTED_PDF_EXTENSIONS = {".pdf"}

# Minimum word count to set the has_text flag.
# Lowered to 1 so sparse dark-mode UI screenshots still qualify.
MIN_WORD_COUNT_FOR_TEXT = 1


class ImageProcessorError(Exception):
    pass


class ImageProcessor:
    """Extract text and visual features from images and PDFs.

    Parameters
    ----------
    tesseract_cmd:
        Optional absolute path to the ``tesseract`` binary.
    ocr_lang:
        Tesseract language code(s), e.g. ``"eng"`` or ``"eng+fra"``.
    min_word_count:
        Minimum number of words OCR must extract for ``has_text`` to be
        ``True``.  Defaults to 3 so that sparse dark-mode screenshots
        still qualify.
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
            raise ImageProcessorError(f"cv2.SIFT_create() is unavailable: {exc}") from exc

    # ------------------------------------------------------------------ #
    # Reading helpers
    # ------------------------------------------------------------------ #
    def _read_grayscale(self, image_path: str) -> np.ndarray:
        if not os.path.isfile(image_path):
            raise ImageProcessorError(f"Image file not found: {image_path}")
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ImageProcessorError(f"OpenCV could not decode image: {image_path}")
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
        if ImageProcessor._is_dark_background(work):
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
        except Exception as exc:
            logger.warning("OCR failed: %s", exc)
            return ""
        return " ".join(raw.split()).strip()

    def _ocr_multi_pass(self, gray: np.ndarray) -> str:
        """Run multi-pass OCR: original + preprocessed, keep best result."""
        # Pass 1: original grayscale
        text_original = self._ocr_single(gray)

        # Pass 2: preprocessed for dark/low-contrast backgrounds
        try:
            preprocessed = self._preprocess_for_ocr(gray)
            text_preprocessed = self._ocr_single(preprocessed)
        except cv2.error as exc:
            logger.debug("Preprocessing failed: %s", exc)
            text_preprocessed = ""

        if len(text_preprocessed) > len(text_original):
            logger.debug(
                "Preprocessed OCR yielded more text (%d vs %d chars)",
                len(text_preprocessed), len(text_original),
            )
            return text_preprocessed
        return text_original

    # ------------------------------------------------------------------ #
    # OCR entry points
    # ------------------------------------------------------------------ #
    def extract_text_from_image(self, image_path: str) -> str:
        """Extract text from a standard image using multi-pass OCR."""
        gray = self._read_grayscale(image_path)
        return self._ocr_multi_pass(gray)

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF by converting pages to images and running OCR."""
        text = []
        try:
            doc = fitz.open(pdf_path)
            for page in doc:
                pix = page.get_pixmap()
                # Convert fitz pixmap to numpy array for OpenCV/Tesseract
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                if pix.n == 4:
                    gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
                elif pix.n == 3:
                    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                else:
                    gray = img

                # Multi-pass OCR on each page
                page_text = self._ocr_multi_pass(gray)
                text.append(page_text)
        except Exception as exc:
            logger.warning("PDF processing failed on %s: %s", pdf_path, exc)
            return ""

        return " ".join(text).strip()

    def extract_text(self, file_path: str) -> str:
        ext = os.path.splitext(file_path)[1].lower()
        if ext in SUPPORTED_PDF_EXTENSIONS:
            return self.extract_text_from_pdf(file_path)
        return self.extract_text_from_image(file_path)

    # ------------------------------------------------------------------ #
    # Template matching
    # ------------------------------------------------------------------ #
    def match_template(self, image_path: str, template_path: str) -> int:
        """
        Identify patterns or logos by matching a template to the image using SIFT.
        Returns the number of good matches found.
        """
        try:
            img1 = self._read_grayscale(template_path) # queryImage
            img2 = self._read_grayscale(image_path)    # trainImage
        except ImageProcessorError as exc:
            logger.warning("Template matching read error: %s", exc)
            return 0

        kp1, des1 = self._sift.detectAndCompute(img1, None)
        kp2, des2 = self._sift.detectAndCompute(img2, None)

        if des1 is None or des2 is None:
            return 0

        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)

        matcher = cv2.FlannBasedMatcher(index_params, search_params)
        matches = matcher.knnMatch(des1, des2, k=2)

        # Apply ratio test
        good_matches = 0
        for match in matches:
            if len(match) == 2:
                m, n = match
                if m.distance < 0.7 * n.distance:
                    good_matches += 1

        return good_matches

    # ------------------------------------------------------------------ #
    # Feature extraction
    # ------------------------------------------------------------------ #
    def extract_features(self, file_path: str) -> int:
        """Return the number of SIFT keypoints detected in the image/pdf."""
        ext = os.path.splitext(file_path)[1].lower()
        if ext in SUPPORTED_PDF_EXTENSIONS:
            try:
                doc = fitz.open(file_path)
                if len(doc) > 0:
                    pix = doc[0].get_pixmap()
                    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                    if pix.n == 4:
                        img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
                    elif pix.n == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    keypoints, _ = self._sift.detectAndCompute(img, None)
                    return len(keypoints) if keypoints is not None else 0
                return 0
            except Exception:
                return 0

        try:
            image = self._read_grayscale(file_path)
            keypoints, _ = self._sift.detectAndCompute(image, None)
            return len(keypoints) if keypoints is not None else 0
        except Exception:
            return 0

    # ------------------------------------------------------------------ #
    # Orchestration
    # ------------------------------------------------------------------ #
    def process_file(self, file_path: str, mime_type: Optional[str] = None) -> dict[str, Any]:
        """Run OCR + SIFT on file_path and return JSON-safe metadata."""
        if not file_path:
            raise ValueError("file_path must be a non-empty string")

        extension = os.path.splitext(file_path)[1].lower()

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
