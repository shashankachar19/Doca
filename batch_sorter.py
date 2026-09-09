"""
DoCA - Document Classification and Analysis
Batch sorter.

Provides :class:`DocumentSorter` which classifies every file in an input
directory by true MIME type (not extension), runs it through the matching
ML handler, persists the metadata to CouchDB, and then physically moves
each file into a category folder under ``output_dir``. Categories with
fewer than ``threshold`` members are consolidated into ``Others``.

Designed to be imported and triggered by the upcoming PyQt6 GUI::

    from batch_sorter import DocumentSorter

    sorter = DocumentSorter()
    result = sorter.sort_directory("/path/to/input", "/path/to/output", threshold=3)
    print(result["tallies"])
"""

from __future__ import annotations

import logging
import os
import shutil
from datetime import datetime
from typing import Any, Callable, Optional

import magic

from handlers.AudioClassifier import AudioClassifier
from handlers.db_handler import DBHandler, DBHandlerError
from handlers.image_handler import ImageHandler
from handlers.text_handler import TextHandler
from handlers.VideoClassifier import VideoClassifier

logger = logging.getLogger("doca.batch_sorter")

# Category labels used both for tallies and for output subfolder names.
CATEGORY_TEXT = "Text"
CATEGORY_IMAGE = "Image"
CATEGORY_AUDIO_MUSIC = "Audio_Music"
CATEGORY_AUDIO_SPEECH = "Audio_Speech"
CATEGORY_GENERAL_VIDEO = "General_Video"
CATEGORY_SECURITY = "Security_Footage"
CATEGORY_OTHERS = "Others"

# Optional progress-callback signature: (current, total, file_path, category).
ProgressCallback = Callable[[int, int, str, Optional[str]], None]

# Generic MIME types libmagic falls back to when the file is a ZIP/CFBF
# container (Office on Windows especially) or a random blob. When we see
# one of these we consult the extension as a second opinion.
_GENERIC_MIME_TYPES = frozenset({
    "application/zip",
    "application/x-zip",
    "application/x-zip-compressed",
    "application/octet-stream",
    "application/x-ole-storage",
    "application/CDFV2",
    "application/vnd.ms-office",
})

# Extension -> top-level handler category used by the smart fallback.
_EXTENSION_CATEGORY: dict[str, str] = {
    # text / documents
    ".txt": "text", ".md": "text", ".rst": "text", ".log": "text",
    ".csv": "text", ".json": "text", ".xml": "text", ".html": "text",
    ".htm": "text", ".pdf": "text", ".rtf": "text",
    ".docx": "text", ".doc": "text",
    ".xlsx": "text", ".xlsm": "text", ".xls": "text",
    ".pptx": "text", ".pptm": "text", ".ppt": "text",
    ".odt": "text", ".ods": "text", ".odp": "text",
    # images
    ".png": "image", ".jpg": "image", ".jpeg": "image", ".bmp": "image",
    ".tif": "image", ".tiff": "image", ".webp": "image", ".gif": "image",
    # audio
    ".mp3": "audio", ".wav": "audio", ".flac": "audio", ".ogg": "audio",
    ".m4a": "audio", ".aac": "audio", ".wma": "audio",
    # video
    ".mp4": "video", ".avi": "video", ".mov": "video", ".mkv": "video",
    ".webm": "video", ".m4v": "video",
}


def _classify_by_extension(file_path: str) -> Optional[str]:
    """Smart fallback: map a file extension to a top-level handler category."""
    ext = os.path.splitext(file_path)[1].lower()
    return _EXTENSION_CATEGORY.get(ext)


def _classify_mime(mime_type: str) -> Optional[str]:
    """Map a libmagic MIME string to a top-level handler category.

    Returns one of {"text", "image", "audio", "video"} or ``None`` if
    the MIME type is unsupported.
    """
    if not mime_type:
        return None

    mime = mime_type.lower().strip()

    if mime.startswith("text/"):
        return "text"
    if mime.startswith("image/"):
        return "image"
    if mime.startswith("audio/"):
        return "audio"
    if mime.startswith("video/"):
        return "video"

    # A handful of application/* types are really text/document content.
    # This whitelist includes Microsoft Office (OOXML + legacy binary),
    # OpenDocument, and common plain-text containers. Everything here is
    # routed to the TextHandler.
    text_application_types = {
        # Generic text-ish payloads
        "application/json",
        "application/xml",
        "application/x-yaml",
        "application/x-sh",
        "application/javascript",
        "application/x-tex",
        "application/x-latex",
        "application/csv",
        "application/pdf",
        "application/rtf",
        "application/x-rtf",

        # Microsoft Office - OOXML (.docx / .xlsx / .pptx)
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",

        # Microsoft Office - legacy binary (.doc / .xls / .ppt)
        "application/msword",
        "application/vnd.ms-word",
        "application/vnd.ms-excel",
        "application/vnd.ms-powerpoint",

        # Microsoft Office - CFBF container reported by libmagic for old .doc/.xls/.ppt
        "application/x-ole-storage",
        "application/vnd.ms-office",
        "application/CDFV2",

        # OpenDocument counterparts
        "application/vnd.oasis.opendocument.text",
        "application/vnd.oasis.opendocument.spreadsheet",
        "application/vnd.oasis.opendocument.presentation",
    }
    if mime in text_application_types:
        return "text"

    # Some containers are reported as application/* but carry media.
    media_application_types = {
        "application/ogg": "audio",
        "application/mp4": "video",
    }
    if mime in media_application_types:
        return media_application_types[mime]

    return None


def _resolve_collision(destination: str) -> str:
    """Return a non-colliding path, appending a timestamp/counter if needed."""
    if not os.path.exists(destination):
        return destination

    directory, filename = os.path.split(destination)
    stem, ext = os.path.splitext(filename)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    candidate = os.path.join(directory, f"{stem}_{timestamp}{ext}")
    counter = 1
    while os.path.exists(candidate):
        candidate = os.path.join(directory, f"{stem}_{timestamp}_{counter}{ext}")
        counter += 1
    return candidate


def _safe_move(src: str, dest_dir: str) -> Optional[str]:
    """Move ``src`` into ``dest_dir`` with collision handling."""
    try:
        os.makedirs(dest_dir, exist_ok=True)
    except OSError as exc:
        logger.error("Could not create destination '%s': %s", dest_dir, exc)
        return None

    target = _resolve_collision(os.path.join(dest_dir, os.path.basename(src)))
    try:
        return shutil.move(src, target)
    except (shutil.Error, OSError) as exc:
        logger.error("Failed to move %s -> %s: %s", src, target, exc)
        return None


class DocumentSorter:
    """Classify, persist and physically sort a directory of files.

    Parameters
    ----------
    db_handler, text_handler, image_handler, audio_handler, video_handler:
        Pre-built handlers. Any not provided will be constructed with
        default settings, which is the common case for GUI users.
    """

    def __init__(
        self,
        db_handler: Optional[DBHandler] = None,
        text_handler: Optional[TextHandler] = None,
        image_handler: Optional[ImageHandler] = None,
        audio_handler: Optional[AudioClassifier] = None,
        video_handler: Optional[VideoClassifier] = None,
    ) -> None:
        self.db = db_handler or DBHandler()
        self.text_handler = text_handler or TextHandler()
        self.image_handler = image_handler or ImageHandler()
        self.audio_handler = audio_handler or AudioClassifier()
        self.video_handler = video_handler or VideoClassifier()

        try:
            self._mime = magic.Magic(mime=True)
        except magic.MagicException as exc:  # type: ignore[attr-defined]
            raise RuntimeError(
                "libmagic is not available. Install 'libmagic' (Linux/macOS) "
                "or 'python-magic-bin' (Windows) and retry."
            ) from exc

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def sort_directory(
        self,
        input_dir: str,
        output_dir: str,
        threshold: int = 3,
        recursive: bool = False,
        progress_cb: Optional[ProgressCallback] = None,
    ) -> dict[str, Any]:
        """Classify and sort every file in ``input_dir``.

        Parameters
        ----------
        input_dir:
            Folder to scan.
        output_dir:
            Folder under which category subfolders will be created.
        threshold:
            Categories with fewer than this many files are collapsed
            into ``Others`` during the physical move.
        recursive:
            When True, walk ``input_dir`` recursively.
        progress_cb:
            Optional callback invoked after each file is processed.
            Signature: ``(index, total, file_path, category_or_None)``.

        Returns
        -------
        dict
            A report with keys:
              * ``tallies``: category -> count
              * ``processed``: list of per-file records
              * ``skipped``: list of unsupported / failed files
              * ``moved``: list of (src, dest, category) tuples
        """
        if not os.path.isdir(input_dir):
            raise NotADirectoryError(f"input_dir is not a directory: {input_dir}")
        if threshold < 1:
            raise ValueError("threshold must be >= 1")

        input_dir = os.path.abspath(input_dir)
        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        if os.path.commonpath([input_dir, output_dir]) == input_dir:
            raise ValueError(
                "output_dir must not live inside input_dir "
                f"(input={input_dir}, output={output_dir})"
            )

        files = list(self._iter_files(input_dir, recursive=recursive))
        logger.info("Found %d files under %s", len(files), input_dir)

        processed, skipped, tallies = self._process_pass(files, progress_cb)
        moved = self._route_pass(processed, output_dir, threshold)

        return {
            "tallies": tallies,
            "processed": processed,
            "skipped": skipped,
            "moved": moved,
            "threshold": threshold,
            "output_dir": output_dir,
        }


    # ------------------------------------------------------------------ #
    # Pass 1: classify, analyze, persist
    # ------------------------------------------------------------------ #
    def _iter_files(self, input_dir: str, recursive: bool):
        """Yield absolute paths of regular files under ``input_dir``."""
        if recursive:
            for root, _dirs, names in os.walk(input_dir):
                for name in names:
                    yield os.path.join(root, name)
        else:
            for name in os.listdir(input_dir):
                full = os.path.join(input_dir, name)
                if os.path.isfile(full):
                    yield full

    def _detect_mime(self, file_path: str) -> Optional[str]:
        """Return libmagic MIME for ``file_path`` or ``None`` on failure."""
        try:
            return self._mime.from_file(file_path)
        except (OSError, magic.MagicException) as exc:  # type: ignore[attr-defined]
            logger.warning("libmagic failed on %s: %s", file_path, exc)
            return None

    def _handler_for(self, category: str):
        """Return the ML handler for a top-level category."""
        return {
            "text": self.text_handler,
            "image": self.image_handler,
            "audio": self.audio_handler,
            "video": self.video_handler,
        }.get(category)

    def _category_for_record(
        self, top_category: str, metadata: dict[str, Any]
    ) -> str:
        """Map a processed record to a display category used for tallies."""
        if top_category == "text":
            return CATEGORY_TEXT
        if top_category == "image":
            return CATEGORY_IMAGE
        if top_category == "audio":
            music_secs = metadata.get("music_seconds", 0)
            speech_secs = metadata.get("male_seconds", 0) + metadata.get("female_seconds", 0)
            if speech_secs > music_secs:
                return CATEGORY_AUDIO_SPEECH
            return CATEGORY_AUDIO_MUSIC
        if top_category == "video":
            if metadata.get("is_security_footage"):
                return CATEGORY_SECURITY
            return CATEGORY_GENERAL_VIDEO
        return CATEGORY_OTHERS

    def _process_pass(
        self,
        files: list[str],
        progress_cb: Optional[ProgressCallback],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
        """Run MIME detection + handler + DB save for every file."""
        processed: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        tallies: dict[str, int] = {}

        total = len(files)
        for index, file_path in enumerate(files, start=1):
            mime = self._detect_mime(file_path)
            if mime is None:
                skipped.append({"path": file_path, "reason": "mime_detection_failed"})
                self._notify(progress_cb, index, total, file_path, None)
                continue

            top_category = _classify_mime(mime)
            if top_category is None:
                # Smart fallback: libmagic on Windows often reports valid
                # .docx/.pptx/.xlsx as application/zip and unknown blobs as
                # application/octet-stream. Consult the extension before
                # giving up so Office files don't get skipped.
                if mime.lower().strip() in _GENERIC_MIME_TYPES:
                    top_category = _classify_by_extension(file_path)
                    if top_category is not None:
                        logger.info(
                            "Smart fallback: libmagic returned generic '%s' "
                            "for %s; routing by extension to '%s' handler.",
                            mime, file_path, top_category,
                        )

            if top_category is None:
                logger.info("Skipping %s (unsupported mime '%s')", file_path, mime)
                skipped.append(
                    {"path": file_path, "reason": "unsupported_mime", "mime": mime}
                )
                self._notify(progress_cb, index, total, file_path, None)
                continue

            handler = self._handler_for(top_category)
            if handler is None:
                skipped.append({"path": file_path, "reason": "no_handler"})
                self._notify(progress_cb, index, total, file_path, None)
                continue

            try:
                metadata: dict[str, Any] = handler.process_file(
                    file_path, mime_type=mime
                )
            except Exception as exc:
                logger.exception("Handler '%s' failed on %s", top_category, file_path)
                # Surface the real exception in the skipped list so the GUI
                # can show e.g. "handler_error: Permission denied" or
                # "handler_error: BadZipFile" instead of a generic label.
                detail = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
                skipped.append(
                    {
                        "path": file_path,
                        "reason": f"handler_error: {detail}",
                        "mime": mime,
                    }
                )
                self._notify(progress_cb, index, total, file_path, None)
                continue

            metadata["mime_type"] = mime
            display_category = self._category_for_record(top_category, metadata)
            metadata["category"] = display_category

            doc_id: Optional[str] = None
            try:
                doc_id = self.db.save_document(metadata)
            except DBHandlerError:
                logger.exception("Failed to persist metadata for %s", file_path)
                # We still sort the file physically; persistence is best-effort.

            tallies[display_category] = tallies.get(display_category, 0) + 1
            processed.append(
                {
                    "path": file_path,
                    "mime": mime,
                    "category": display_category,
                    "doc_id": doc_id,
                    "metadata": metadata,
                }
            )
            logger.info(
                "Processed %s [%s] -> %s (_id=%s)",
                file_path, mime, display_category, doc_id,
            )
            self._notify(progress_cb, index, total, file_path, display_category)

        return processed, skipped, tallies

    @staticmethod
    def _notify(
        cb: Optional[ProgressCallback],
        index: int,
        total: int,
        path: str,
        category: Optional[str],
    ) -> None:
        if cb is None:
            return
        try:
            cb(index, total, path, category)
        except Exception:
            logger.exception("Progress callback raised; continuing.")


    # ------------------------------------------------------------------ #
    # Pass 2: physical routing with threshold
    # ------------------------------------------------------------------ #
    def _route_pass(
        self,
        processed: list[dict[str, Any]],
        output_dir: str,
        threshold: int,
    ) -> list[dict[str, Any]]:
        """Move processed files based on per-category tallies and threshold."""
        # Re-tally from the processed list so this pass is self-contained.
        tallies: dict[str, int] = {}
        for record in processed:
            tallies[record["category"]] = tallies.get(record["category"], 0) + 1

        moved: list[dict[str, Any]] = []
        for record in processed:
            category = record["category"]
            count = tallies.get(category, 0)
            target_folder = category if count >= threshold else CATEGORY_OTHERS
            dest_dir = os.path.join(output_dir, target_folder)

            src = record["path"]
            if not os.path.isfile(src):
                logger.warning("Source vanished before move: %s", src)
                continue

            final_path = _safe_move(src, dest_dir)
            if final_path is None:
                continue

            # Best-effort: record the new physical location in CouchDB.
            doc_id = record.get("doc_id")
            if doc_id:
                try:
                    updated = dict(record["metadata"])
                    updated["_id"] = doc_id
                    updated["stored_path"] = final_path
                    updated["category_folder"] = target_folder
                    self.db.save_document(updated)
                except DBHandlerError:
                    logger.warning(
                        "Moved %s but failed to update its CouchDB record.",
                        final_path,
                    )

            logger.info(
                "Routed %s -> %s (category=%s, count=%d, threshold=%d)",
                src, final_path, category, count, threshold,
            )
            moved.append(
                {
                    "src": src,
                    "dest": final_path,
                    "category": category,
                    "target_folder": target_folder,
                }
            )
        return moved
