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

class ImageProcessorError(Exception):
    pass

class ImageProcessor:
    FILE_TYPE = "image"

    def __init__(self, tesseract_cmd: Optional[str] = None, ocr_lang: str = "eng") -> None:
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        self.ocr_lang = ocr_lang

        try:
            self._sift = cv2.SIFT_create()
        except AttributeError as exc:
            raise ImageProcessorError(f"cv2.SIFT_create() is unavailable: {exc}") from exc

    def _read_grayscale(self, image_path: str) -> np.ndarray:
        if not os.path.isfile(image_path):
            raise ImageProcessorError(f"Image file not found: {image_path}")
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ImageProcessorError(f"OpenCV could not decode image: {image_path}")
        return image

    def extract_text_from_image(self, image_path: str) -> str:
        """Extract text from a standard image using Tesseract OCR."""
        image = self._read_grayscale(image_path)
        try:
            raw = pytesseract.image_to_string(image, lang=self.ocr_lang)
        except Exception as exc:
            logger.warning("OCR failed on %s: %s", image_path, exc)
            return ""
        return " ".join(raw.split()).strip()

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
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
                elif pix.n == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

                raw = pytesseract.image_to_string(img, lang=self.ocr_lang)
                text.append(raw)
        except Exception as exc:
            logger.warning("PDF processing failed on %s: %s", pdf_path, exc)
            return ""

        return " ".join(" ".join(text).split()).strip()

    def extract_text(self, file_path: str) -> str:
        ext = os.path.splitext(file_path)[1].lower()
        if ext in SUPPORTED_PDF_EXTENSIONS:
            return self.extract_text_from_pdf(file_path)
        return self.extract_text_from_image(file_path)

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

    def extract_features(self, file_path: str) -> int:
        """Return the number of SIFT keypoints detected in the image/pdf."""
        ext = os.path.splitext(file_path)[1].lower()
        if ext in SUPPORTED_PDF_EXTENSIONS:
            # We skip feature extraction for PDFs to keep it simple, or we could do it for page 1.
            # Let's just do it for page 1 if needed.
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

    def process_file(self, file_path: str, mime_type: Optional[str] = None) -> dict[str, Any]:
        """Run OCR + SIFT on file_path and return JSON-safe metadata."""
        if not file_path:
            raise ValueError("file_path must be a non-empty string")

        extension = os.path.splitext(file_path)[1].lower()

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
