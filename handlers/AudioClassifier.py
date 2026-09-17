"""Audio file handler for the DoCA project.

Provides an :class:`AudioClassifier` that runs Ina's Speech Segmenter on an
audio file, summarizes the detected labels (music / male / female /
noise / silence) and returns a JSON-safe metadata dictionary suitable
for :meth:`handlers.db_handler.DBHandler.save_document`.

On platforms where ``inaSpeechSegmenter`` is unavailable (Windows, macOS),
the classifier degrades gracefully: audio files are still catalogued with
basic metadata but labelled as ``Audio`` instead of being split into
speech vs. music.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Optional

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".wma"}

# Labels emitted by inaSpeechSegmenter that we explicitly surface.
# Anything else is tracked under "other_seconds".
_TRACKED_LABELS = {"music", "male", "female", "noise", "noEnergy"}

# Whether inaSpeechSegmenter is available on this platform.
_INA_AVAILABLE: bool
try:
    from inaSpeechSegmenter import Segmenter as _Segmenter  # noqa: F401
    _INA_AVAILABLE = True
except ImportError:
    _INA_AVAILABLE = False
    if sys.platform != "linux":
        logger.info(
            "inaSpeechSegmenter is not installed (expected on %s). "
            "Audio classification will use graceful fallback.",
            sys.platform,
        )
    else:
        logger.warning(
            "inaSpeechSegmenter is not installed. "
            "Install it with: pip install inaSpeechSegmenter"
        )


class AudioClassifierError(Exception):
    """Raised when the AudioClassifier cannot complete an operation."""


class AudioClassifier:
    """Segment audio files and summarize time per class.

    Parameters
    ----------
    vad_engine:
        Voice activity detection engine for the segmenter. ``"smn"`` is
        the default and yields music/speech/noise labels.
    detect_gender:
        When ``True`` (default), speech is further split into male/female.
    """

    FILE_TYPE = "audio"

    def __init__(
        self,
        vad_engine: str = "smn",
        detect_gender: bool = True,
    ) -> None:
        self.vad_engine = vad_engine
        self.detect_gender = detect_gender
        self._segmenter: Optional[Any] = None
        self._ina_available = _INA_AVAILABLE
        # The segmenter downloads CNN weights on first construction, so
        # we lazily build it the first time it's actually needed.

    @property
    def is_available(self) -> bool:
        """Return True if full speech/music segmentation is available."""
        return self._ina_available

    # ------------------------------------------------------------------ #
    # Lazy segmenter init
    # ------------------------------------------------------------------ #
    def _get_segmenter(self) -> Any:
        """Return the cached Segmenter, initializing it on first use."""
        if self._segmenter is not None:
            return self._segmenter

        if not self._ina_available:
            raise AudioClassifierError(
                "inaSpeechSegmenter is not installed on this platform. "
                "Audio classification is running in fallback mode."
            )

        try:
            from inaSpeechSegmenter import Segmenter
        except ImportError as exc:
            self._ina_available = False
            raise AudioClassifierError(
                "inaSpeechSegmenter is not installed. "
                "Add it to requirements.txt and reinstall."
            ) from exc

        try:
            self._segmenter = Segmenter(
                vad_engine=self.vad_engine,
                detect_gender=self.detect_gender,
            )
        except Exception as exc:  # TF / HDF5 / network errors on weight fetch
            raise AudioClassifierError(
                "Failed to initialize inaSpeechSegmenter "
                f"(check TensorFlow install and network access for model download): {exc}"
            ) from exc

        return self._segmenter

    # ------------------------------------------------------------------ #
    # Segmentation
    # ------------------------------------------------------------------ #
    def segment_audio(self, audio_path: str) -> list[tuple[str, float, float]]:
        """Run the segmenter on ``audio_path``.

        Returns a list of ``(label, start_seconds, end_seconds)`` tuples
        as produced by inaSpeechSegmenter.
        """
        if not audio_path:
            raise ValueError("audio_path must be a non-empty string")
        if not os.path.isfile(audio_path):
            raise AudioClassifierError(f"Audio file not found: {audio_path}")

        segmenter = self._get_segmenter()

        try:
            segments = segmenter(audio_path)
        except FileNotFoundError as exc:
            # Typically raised when ffmpeg is missing from PATH.
            raise AudioClassifierError(
                f"Audio decoding failed. Ensure ffmpeg is installed and on PATH: {exc}"
            ) from exc
        except Exception as exc:
            raise AudioClassifierError(
                f"Segmentation failed for '{audio_path}': {exc}"
            ) from exc

        # Normalize the returned rows to plain Python tuples/floats so
        # downstream code (and JSON serialization) stays simple.
        normalized: list[tuple[str, float, float]] = []
        for row in segments:
            try:
                label, start, end = row
            except (TypeError, ValueError):
                logger.debug("Skipping unexpected segmenter row: %r", row)
                continue
            normalized.append((str(label), float(start), float(end)))
        return normalized

    # ------------------------------------------------------------------ #
    # Summarization
    # ------------------------------------------------------------------ #
    @staticmethod
    def _summarize(
        segments: list[tuple[str, float, float]],
    ) -> dict[str, Any]:
        """Aggregate per-label durations from a list of segments."""
        totals: dict[str, float] = {}
        for label, start, end in segments:
            duration = max(0.0, end - start)
            totals[label] = totals.get(label, 0.0) + duration

        music = totals.get("music", 0.0)
        male = totals.get("male", 0.0)
        female = totals.get("female", 0.0)
        noise = totals.get("noise", 0.0)
        silence = totals.get("noEnergy", 0.0)
        other = sum(
            d for label, d in totals.items() if label not in _TRACKED_LABELS
        )
        total = sum(totals.values())
        speech = male + female

        def pct(value: float) -> Optional[float]:
            if total <= 0:
                return None
            return round((value / total) * 100.0, 2)

        return {
            "music_seconds": round(music, 3),
            "male_seconds": round(male, 3),
            "female_seconds": round(female, 3),
            "speech_seconds": round(speech, 3),
            "noise_seconds": round(noise, 3),
            "silence_seconds": round(silence, 3),
            "other_seconds": round(other, 3),
            "total_seconds": round(total, 3),
            "music_pct": pct(music),
            "speech_pct": pct(speech),
        }

    # ------------------------------------------------------------------ #
    # Fallback metadata (when inaSpeechSegmenter is unavailable)
    # ------------------------------------------------------------------ #
    def _fallback_metadata(self, file_path: str) -> dict[str, Any]:
        """Return basic metadata when segmentation is not available."""
        try:
            size_bytes = os.path.getsize(file_path)
        except OSError:
            size_bytes = None

        return {
            "path": os.path.abspath(file_path),
            "file_name": os.path.basename(file_path),
            "file_type": self.FILE_TYPE,
            "extension": os.path.splitext(file_path)[1].lower(),
            "size_bytes": size_bytes,
            "segment_count": 0,
            "segments": [],
            "music_seconds": 0.0,
            "male_seconds": 0.0,
            "female_seconds": 0.0,
            "speech_seconds": 0.0,
            "noise_seconds": 0.0,
            "silence_seconds": 0.0,
            "other_seconds": 0.0,
            "total_seconds": 0.0,
            "music_pct": None,
            "speech_pct": None,
            "classification_note": (
                "inaSpeechSegmenter is not available on this platform. "
                "Audio was catalogued but not classified into speech/music."
            ),
        }

    # ------------------------------------------------------------------ #
    # Orchestration
    # ------------------------------------------------------------------ #
    def process_file(
        self,
        file_path: str,
        mime_type: Optional[str] = None,
    ) -> dict[str, Any]:
        """Run segmentation on ``file_path`` and return a JSON-safe dict.

        ``mime_type`` is accepted for a uniform content-based call signature
        across all handlers; segmentation is MIME-agnostic, so the hint is
        currently unused.

        When ``inaSpeechSegmenter`` is unavailable, returns basic metadata
        with zero-valued segment summaries instead of raising.
        """
        if not file_path:
            raise ValueError("file_path must be a non-empty string")
        _ = mime_type

        extension = os.path.splitext(file_path)[1].lower()
        if extension and extension not in SUPPORTED_EXTENSIONS:
            logger.debug("Unusual audio extension '%s'; attempting anyway.", extension)

        if not os.path.isfile(file_path):
            raise AudioClassifierError(f"Audio file not found: {file_path}")

        # Graceful degradation: if the segmenter isn't available, return
        # basic metadata so the rest of the pipeline can continue.
        if not self._ina_available:
            logger.info(
                "Returning fallback metadata for %s "
                "(inaSpeechSegmenter not available)",
                file_path,
            )
            return self._fallback_metadata(file_path)

        segments = self.segment_audio(file_path)
        summary = self._summarize(segments)

        try:
            size_bytes = os.path.getsize(file_path)
        except OSError:
            size_bytes = None

        # Keep the raw segment list available but compact and JSON-safe.
        serialized_segments = [
            {
                "label": label,
                "start": round(start, 3),
                "end": round(end, 3),
            }
            for label, start, end in segments
        ]

        return {
            "path": os.path.abspath(file_path),
            "file_name": os.path.basename(file_path),
            "file_type": self.FILE_TYPE,
            "extension": extension,
            "size_bytes": size_bytes,
            "segment_count": len(segments),
            "segments": serialized_segments,
            **summary,
        }
