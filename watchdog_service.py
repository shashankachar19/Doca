"""
DoCA - Document Classification and Analysis
Main watchdog service.

Monitors a target folder for new or modified files, dispatches each one
to the appropriate handler based on extension, and persists the returned
metadata dictionary to CouchDB via :class:`DBHandler`.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import signal
import sys
import threading
import time
from datetime import datetime
from typing import Any, Optional

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from handlers.AudioClassifier import AudioClassifier
from handlers.db_handler import DBHandler, DBHandlerError
from handlers.ImageProcessor import ImageProcessor
from handlers.TextClassifier import TextClassifier
from handlers.VideoClassifier import VideoClassifier

logger = logging.getLogger("doca.watchdog")

DEFAULT_WATCH_DIR = "./monitored_folder"
DEFAULT_OUTPUT_DIR = "./organized_output"

# Subfolder names used under the output directory.
CATEGORY_FOLDERS: dict[str, str] = {
    "text": "Text",
    "image": "Image",
    # audio and video are resolved dynamically
}
AUDIO_MUSIC_FOLDER = "Audio_Music"
AUDIO_SPEECH_FOLDER = "Audio_Speech"

VIDEO_SECURITY_FOLDER = "Security_Footage"
VIDEO_GENERAL_FOLDER = "General_Video"

# Map lower-cased extensions to a logical file category.
EXTENSION_MAP: dict[str, str] = {
    # text
    ".txt": "text", ".md": "text", ".rst": "text", ".log": "text",
    ".csv": "text", ".json": "text", ".xml": "text", ".html": "text",
    ".htm": "text",
    # image
    ".png": "image", ".jpg": "image", ".jpeg": "image", ".bmp": "image",
    ".tif": "image", ".tiff": "image", ".webp": "image", ".gif": "image",
    ".pdf": "image",
    # video
    ".mp4": "video", ".avi": "video", ".mov": "video", ".mkv": "video",
    ".webm": "video", ".m4v": "video",
    # audio
    ".mp3": "audio", ".wav": "audio", ".flac": "audio", ".ogg": "audio",
    ".m4a": "audio", ".aac": "audio", ".wma": "audio",
}

# Files that should never be processed, even if matched by extension.
IGNORED_PREFIXES = (".", "~")
IGNORED_SUFFIXES = (".tmp", ".part", ".crdownload", ".swp")


def classify_extension(file_path: str) -> Optional[str]:
    """Return the logical category for ``file_path`` or ``None``."""
    ext = os.path.splitext(file_path)[1].lower()
    return EXTENSION_MAP.get(ext)


def _should_ignore(file_path: str) -> bool:
    """True for hidden, temporary, or partial-download files."""
    name = os.path.basename(file_path)
    if not name:
        return True
    if name.startswith(IGNORED_PREFIXES):
        return True
    if name.endswith(IGNORED_SUFFIXES):
        return True
    return False


def _wait_until_stable(
    file_path: str,
    poll_interval: float = 0.5,
    max_wait: float = 30.0,
) -> bool:
    """Wait until ``file_path``'s size stops changing.

    Large files (video, audio) often trigger ``on_created`` before the
    writer has finished. We poll ``os.path.getsize`` until two successive
    reads match, then return True. Returns False if the file disappears
    or the timeout is reached.
    """
    deadline = time.monotonic() + max_wait
    last_size = -1
    while time.monotonic() < deadline:
        try:
            size = os.path.getsize(file_path)
        except OSError:
            return False
        if size == last_size and size > 0:
            return True
        last_size = size
        time.sleep(poll_interval)
    # Fall through: treat as stable enough to attempt processing.
    return os.path.isfile(file_path)


def _resolve_collision(destination: str) -> str:
    """Return a non-colliding destination path.

    If ``destination`` already exists, a timestamp (and, if needed, an
    incrementing counter) is inserted before the extension.
    """
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
    """Move ``src`` into ``dest_dir``, handling collisions. Return new path."""
    try:
        os.makedirs(dest_dir, exist_ok=True)
    except OSError as exc:
        logger.error("Could not create destination '%s': %s", dest_dir, exc)
        return None

    target = _resolve_collision(os.path.join(dest_dir, os.path.basename(src)))
    try:
        final_path = shutil.move(src, target)
    except (shutil.Error, OSError) as exc:
        logger.error("Failed to move %s -> %s: %s", src, target, exc)
        return None
    return final_path


class DoCAEventHandler(FileSystemEventHandler):
    """Route file-system events to the right DoCA handler."""

    def __init__(
        self,
        db_handler: DBHandler,
        text_handler: TextClassifier,
        image_handler: ImageProcessor,
        video_handler: VideoClassifier,
        audio_handler: AudioClassifier,
        output_dir: str = DEFAULT_OUTPUT_DIR,
    ) -> None:
        super().__init__()
        self.db = db_handler
        self.handlers = {
            "text": text_handler,
            "image": image_handler,
            "video": video_handler,
            "audio": audio_handler,
        }
        self.output_dir = os.path.abspath(output_dir)
        # Guard against double-processing the same path when both
        # on_created and on_modified fire in quick succession.
        self._inflight: set[str] = set()
        self._inflight_lock = threading.Lock()

    # Watchdog event hooks -------------------------------------------- #
    def on_created(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        self._process_event(event.src_path, "created")

    def on_modified(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        self._process_event(event.src_path, "modified")

    # Internal -------------------------------------------------------- #
    def _process_event(self, file_path: str, reason: str) -> None:
        path = os.path.abspath(file_path)

        if _should_ignore(path):
            logger.debug("Ignoring %s (hidden/temp file)", path)
            return

        category = classify_extension(path)
        if category is None:
            logger.info("Skipping %s (unsupported extension)", path)
            return

        with self._inflight_lock:
            if path in self._inflight:
                logger.debug("Already processing %s; skipping %s event", path, reason)
                return
            self._inflight.add(path)

        try:
            logger.info("Detected %s: %s (category=%s)", reason, path, category)
            if not _wait_until_stable(path):
                logger.warning("File vanished or never stabilized: %s", path)
                return
            self._dispatch(path, category)
        finally:
            with self._inflight_lock:
                self._inflight.discard(path)

    def _dispatch(self, file_path: str, category: str) -> None:
        handler = self.handlers.get(category)
        if handler is None:
            logger.warning("No handler registered for category '%s'", category)
            return

        try:
            metadata: dict[str, Any] = handler.process_file(file_path)
        except Exception:
            # Never let a single bad file kill the service.
            logger.exception("Handler '%s' failed on %s", category, file_path)
            return

        try:
            doc_id = self.db.save_document(metadata)
        except DBHandlerError:
            logger.exception("Failed to persist metadata for %s", file_path)
            return
        except Exception:
            logger.exception("Unexpected error while saving %s", file_path)
            return

        logger.info(
            "Saved %s metadata for %s (_id=%s)", category, file_path, doc_id
        )

        # Physical routing happens only after a successful DB write so
        # we never end up with a file in the output tree that CouchDB
        # doesn't know about.
        self._route_file(file_path, category, metadata, doc_id)

    def _resolve_subfolder(
        self, category: str, metadata: dict[str, Any]
    ) -> str:
        """Pick the output subfolder name for a given category/result."""
        if category == "video":
            if metadata.get("is_security_footage"):
                return VIDEO_SECURITY_FOLDER
            return VIDEO_GENERAL_FOLDER
        if category == "audio":
            music_secs = metadata.get("music_seconds", 0)
            speech_secs = metadata.get("male_seconds", 0) + metadata.get("female_seconds", 0)
            if speech_secs > music_secs:
                return AUDIO_SPEECH_FOLDER
            return AUDIO_MUSIC_FOLDER
        return CATEGORY_FOLDERS.get(category, category.capitalize())

    def _route_file(
        self,
        file_path: str,
        category: str,
        metadata: dict[str, Any],
        doc_id: str,
    ) -> None:
        """Move the processed file into its category subfolder."""
        subfolder = self._resolve_subfolder(category, metadata)
        dest_dir = os.path.join(self.output_dir, subfolder)

        # Don't shoot ourselves in the foot if someone points the
        # watcher at the output tree.
        try:
            if os.path.commonpath([os.path.abspath(file_path), dest_dir]) == dest_dir:
                logger.debug("File %s is already inside output; skipping move.", file_path)
                return
        except ValueError:
            # commonpath raises for paths on different drives (Windows).
            pass

        final_path = _safe_move(file_path, dest_dir)
        if final_path is None:
            return

        logger.info(
            "Routed %s -> %s (category=%s, _id=%s)",
            file_path, final_path, category, doc_id,
        )

        # Best-effort: update the CouchDB record with the new location.
        try:
            updated = dict(metadata)
            updated["_id"] = doc_id
            updated["stored_path"] = final_path
            updated["category_folder"] = subfolder
            self.db.save_document(updated)
        except DBHandlerError:
            logger.warning(
                "Moved %s but failed to update its CouchDB record.", final_path
            )


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _build_event_handler(output_dir: str) -> DoCAEventHandler:
    """Instantiate all handlers. Fails fast if CouchDB is unreachable."""
    logger.info("Initializing DoCA handlers...")
    db_handler = DBHandler()
    text_handler = TextClassifier()
    image_handler = ImageProcessor()
    video_handler = VideoClassifier()
    audio_handler = AudioClassifier()
    logger.info("Handlers ready.")
    return DoCAEventHandler(
        db_handler=db_handler,
        text_handler=text_handler,
        image_handler=image_handler,
        video_handler=video_handler,
        audio_handler=audio_handler,
        output_dir=output_dir,
    )


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DoCA watchdog service: classify files as they appear."
    )
    parser.add_argument(
        "--path",
        default=DEFAULT_WATCH_DIR,
        help=f"Directory to watch (default: {DEFAULT_WATCH_DIR})",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to route files into (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recurse into subdirectories.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    _configure_logging(args.log_level)

    watch_dir = os.path.abspath(args.path)
    os.makedirs(watch_dir, exist_ok=True)
    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir, exist_ok=True)
    logger.info("Watching directory: %s (recursive=%s)", watch_dir, args.recursive)
    logger.info("Routing processed files into: %s", output_dir)

    if os.path.commonpath([watch_dir, output_dir]) in (watch_dir, output_dir):
        logger.error(
            "Watch directory and output directory must not be nested "
            "(watch=%s, output=%s).",
            watch_dir, output_dir,
        )
        return 2

    try:
        event_handler = _build_event_handler(output_dir)
    except Exception:
        logger.exception("Failed to initialize DoCA handlers. Aborting.")
        return 1

    observer = Observer()
    observer.schedule(event_handler, watch_dir, recursive=args.recursive)
    observer.start()
    logger.info("DoCA watchdog started. Press Ctrl+C to stop.")

    stop_event = threading.Event()

    def _handle_signal(signum: int, _frame: Any) -> None:
        logger.info("Received signal %s; shutting down...", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        while not stop_event.is_set():
            stop_event.wait(timeout=1.0)
    finally:
        observer.stop()
        observer.join(timeout=10)
        logger.info("DoCA watchdog stopped.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
