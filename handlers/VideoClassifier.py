"""Video file handler for the DoCA project.

Provides a :class:`VideoClassifier` that samples frames from a video at
~1 FPS, computes the average structural similarity (SSIM) between
consecutive frames, and uses that ratio combined with a secondary
motion-spread analysis (``cv2.absdiff``) as a heuristic for whether
the video is likely stationary security-camera footage.

Classification strategy
-----------------------
Live wallpapers have high global SSIM (>0.90) because the background is
static, but they exhibit *localised* looping motion (particles, waves).
Real security footage has near-identical frames with virtually zero
pixel change anywhere.  We therefore require **two** conditions:

1. **Global SSIM ≥ 0.98** — tightened from the old 0.85 to exclude
   wallpapers whose SSIM typically lands in the 0.88–0.96 range.
2. **Mean motion contour ratio ≤ 5 %** — the fraction of the frame area
   covered by changed-pixel contours (via ``cv2.absdiff`` + threshold +
   ``findContours``).  Security feeds sit well below 2 %; wallpapers
   with even tiny animations reach 5–15 %.

Both conditions must be true for the ``is_security_footage`` flag.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import cv2
import numpy as np
from skimage.metrics import structural_similarity

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

# Default thresholds — can be overridden via constructor.
DEFAULT_SSIM_THRESHOLD = 0.98
DEFAULT_MOTION_CONTOUR_THRESHOLD = 0.05  # 5 % of frame area


class VideoClassifierError(Exception):
    """Raised when the VideoClassifier cannot complete an operation."""


class VideoClassifier:
    """Analyze videos to flag likely stationary security-camera footage.

    Parameters
    ----------
    target_height:
        Height in pixels used when downscaling each sampled frame.
        Aspect ratio is preserved. Defaults to 144 (144p).
    security_threshold:
        Mean SSIM above which the video *may* be security footage.
        Raised to 0.98 to exclude live wallpapers that typically
        land between 0.88–0.96.
    motion_contour_threshold:
        Maximum fraction of frame area covered by motion contours for
        the video to be considered security footage.  Defaults to 0.05
        (5 %).
    sample_fps:
        Frames per second to sample from the source video. Defaults to 1.
    """

    FILE_TYPE = "video"

    def __init__(
        self,
        target_height: int = 144,
        security_threshold: float = DEFAULT_SSIM_THRESHOLD,
        motion_contour_threshold: float = DEFAULT_MOTION_CONTOUR_THRESHOLD,
        sample_fps: float = 1.0,
    ) -> None:
        if target_height <= 0:
            raise ValueError("target_height must be positive")
        if sample_fps <= 0:
            raise ValueError("sample_fps must be positive")
        self.target_height = target_height
        self.security_threshold = security_threshold
        self.motion_contour_threshold = motion_contour_threshold
        self.sample_fps = sample_fps

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _open_capture(self, video_path: str) -> cv2.VideoCapture:
        """Open a VideoCapture or raise VideoClassifierError."""
        if not os.path.isfile(video_path):
            raise VideoClassifierError(f"Video file not found: {video_path}")

        try:
            cap = cv2.VideoCapture(video_path)
        except cv2.error as exc:
            raise VideoClassifierError(
                f"OpenCV failed to open '{video_path}': {exc}"
            ) from exc

        if not cap.isOpened():
            cap.release()
            raise VideoClassifierError(
                f"OpenCV could not open '{video_path}' "
                "(unsupported codec or corrupted file)."
            )
        return cap

    def _resize_to_target(self, frame: np.ndarray) -> np.ndarray:
        """Resize ``frame`` to ``self.target_height`` keeping aspect ratio."""
        height, width = frame.shape[:2]
        if height == 0 or width == 0:
            raise VideoClassifierError("Frame has zero dimensions")
        new_h = self.target_height
        new_w = max(1, int(round(width * (new_h / height))))
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # ------------------------------------------------------------------ #
    # Frame extraction
    # ------------------------------------------------------------------ #
    def _extract_frames(self, video_path: str) -> list[np.ndarray]:
        """Return grayscale frames sampled at ~``self.sample_fps`` per second.

        Each frame is converted to grayscale and resized to the target
        height while preserving the aspect ratio.
        """
        cap = self._open_capture(video_path)
        frames: list[np.ndarray] = []

        try:
            source_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            if source_fps <= 0 or np.isnan(source_fps):
                logger.warning(
                    "Invalid FPS reported for %s; falling back to 25.", video_path
                )
                source_fps = 25.0

            # Sample every Nth frame to approximate sample_fps.
            step = max(1, int(round(source_fps / self.sample_fps)))

            frame_index = 0
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break

                if frame_index % step == 0:
                    try:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        resized = self._resize_to_target(gray)
                        frames.append(resized)
                    except cv2.error as exc:
                        logger.warning(
                            "Skipping frame %d in %s: %s",
                            frame_index, video_path, exc,
                        )

                frame_index += 1
        finally:
            cap.release()

        return frames

    # ------------------------------------------------------------------ #
    # SSIM
    # ------------------------------------------------------------------ #
    def _calculate_ssim(self, frames: list[np.ndarray]) -> Optional[float]:
        """Return the mean SSIM over consecutive frame pairs.

        Returns ``None`` when fewer than two frames are available.
        """
        if len(frames) < 2:
            return None

        scores: list[float] = []
        for prev, curr in zip(frames, frames[1:]):
            if prev.shape != curr.shape:
                # Shouldn't happen given uniform resizing, but guard anyway.
                logger.debug(
                    "Skipping SSIM pair with mismatched shapes %s vs %s",
                    prev.shape, curr.shape,
                )
                continue
            try:
                score = structural_similarity(prev, curr, data_range=255)
            except ValueError as exc:
                # e.g. frames smaller than the default 7x7 window.
                logger.warning("SSIM failed on a frame pair: %s", exc)
                continue
            scores.append(float(score))

        if not scores:
            return None
        return float(sum(scores) / len(scores))

    # ------------------------------------------------------------------ #
    # Motion-spread analysis (absdiff + contour area)
    # ------------------------------------------------------------------ #
    def _calculate_motion_contour_ratio(
        self, frames: list[np.ndarray]
    ) -> Optional[float]:
        """Return the mean fraction of the frame covered by motion contours.

        For each consecutive pair of sampled frames:

        1. Compute the per-pixel absolute difference (``cv2.absdiff``).
        2. Gaussian-blur the diff to suppress sensor noise.
        3. Binary-threshold the blurred diff (pixel > 15 → motion).
        4. Find external contours and sum their bounding-box areas.
        5. Divide total contour area by total frame area.

        Security footage has near-zero contour ratios (<2 %) because
        nothing moves.  Live wallpapers with even small particle effects
        typically produce 5–15 % contour coverage.

        Returns ``None`` when fewer than two frames are available.
        """
        if len(frames) < 2:
            return None

        ratios: list[float] = []
        for prev, curr in zip(frames, frames[1:]):
            if prev.shape != curr.shape:
                continue
            try:
                diff = cv2.absdiff(prev, curr)
                blurred = cv2.GaussianBlur(diff, (5, 5), 0)
                _, thresh = cv2.threshold(blurred, 15, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(
                    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                frame_area = float(prev.shape[0] * prev.shape[1])
                if frame_area <= 0:
                    continue

                # Sum bounding-rect areas (faster / more stable than
                # cv2.contourArea which can return 0 for very thin contours).
                contour_area = sum(
                    w * h for (_, _, w, h) in (cv2.boundingRect(c) for c in contours)
                )
                ratios.append(contour_area / frame_area)
            except cv2.error as exc:
                logger.debug("Motion analysis failed on a frame pair: %s", exc)
                continue

        if not ratios:
            return None
        return float(sum(ratios) / len(ratios))

    # ------------------------------------------------------------------ #
    # Security footage decision
    # ------------------------------------------------------------------ #
    def _is_security_footage(
        self,
        ssim: Optional[float],
        motion_ratio: Optional[float],
    ) -> bool:
        """Determine security-footage flag from SSIM and motion ratio.

        Both conditions must be met:
        1. Mean SSIM ≥ ``security_threshold`` (default 0.98)
        2. Mean motion-contour ratio ≤ ``motion_contour_threshold`` (default 5 %)

        This dual-gate filters out live wallpapers that pass the SSIM
        test alone due to their mostly-static backgrounds.
        """
        if ssim is None:
            return False

        if ssim < self.security_threshold:
            return False

        # If motion analysis is unavailable (e.g. single-frame video),
        # fall back to SSIM-only with the tightened threshold.
        if motion_ratio is None:
            return True

        return motion_ratio <= self.motion_contour_threshold

    # ------------------------------------------------------------------ #
    # Orchestration
    # ------------------------------------------------------------------ #
    def _probe_metadata(self, video_path: str) -> dict[str, Any]:
        """Read FPS / resolution / frame count without decoding frames."""
        cap = self._open_capture(video_path)
        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        finally:
            cap.release()

        duration_s: Optional[float] = None
        if fps > 0 and frame_count > 0:
            duration_s = round(frame_count / fps, 3)

        return {
            "fps": round(fps, 3) if fps else None,
            "width": width or None,
            "height": height or None,
            "resolution": f"{width}x{height}" if width and height else None,
            "frame_count": frame_count or None,
            "duration_seconds": duration_s,
        }

    def process_file(
        self,
        file_path: str,
        mime_type: Optional[str] = None,
    ) -> dict[str, Any]:
        """Run the full pipeline and return a JSON-safe metadata dict.

        ``mime_type`` is accepted for a uniform content-based call signature
        across all handlers; it is not used to alter the probing/SSIM steps.
        """
        if not file_path:
            raise ValueError("file_path must be a non-empty string")
        _ = mime_type

        extension = os.path.splitext(file_path)[1].lower()
        if extension and extension not in SUPPORTED_EXTENSIONS:
            logger.debug("Unusual video extension '%s'; attempting anyway.", extension)

        probe = self._probe_metadata(file_path)
        frames = self._extract_frames(file_path)
        similarity_ratio = self._calculate_ssim(frames)
        motion_contour_ratio = self._calculate_motion_contour_ratio(frames)
        is_security = self._is_security_footage(similarity_ratio, motion_contour_ratio)

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
            "fps": probe["fps"],
            "width": probe["width"],
            "height": probe["height"],
            "resolution": probe["resolution"],
            "frame_count": probe["frame_count"],
            "duration_seconds": probe["duration_seconds"],
            "sampled_frame_count": len(frames),
            "similarity_ratio": similarity_ratio,
            "motion_contour_ratio": (
                round(motion_contour_ratio, 6) if motion_contour_ratio is not None else None
            ),
            "is_security_footage": is_security,
            "security_threshold": self.security_threshold,
            "motion_contour_threshold": self.motion_contour_threshold,
        }
