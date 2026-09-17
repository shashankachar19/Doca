"""
DoCA - Document Classification and Analysis
Main PyQt6 dashboard.

Three tabs:
  1. Batch Sorter     - run DocumentSorter on a chosen input/output pair.
  2. Live Watchdog    - start/stop the folder watchdog with a live log view.
  3. Database Dashboard - browse processed records from CouchDB.

Architecture notes
------------------
* A single :class:`HandlerPool` lives on the :class:`MainWindow` and lazily
  constructs each ML handler the first time it is requested. The pool is
  shared with every worker so the heavy NLTK / Gensim / OpenCV / Keras models
  only initialize once per application lifetime.
* All long-running work (batch sort, watchdog observer, DB queries) runs on
  QThread workers using the worker-object pattern (QObject moved into a
  QThread). Cross-thread communication uses Qt signals/slots so widget
  mutations always happen on the GUI thread.
* :class:`QtLogHandler` intercepts stdlib ``logging`` records and emits them
  as a Qt signal, letting the Watchdog tab behave like a live terminal.
"""

from __future__ import annotations

import logging
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from typing import Any, Optional

from PyQt6.QtCore import Qt, QObject, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QAction, QTextCursor
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QFileDialog, QFormLayout, QFrame, QHBoxLayout,
    QHeaderView, QLabel, QLineEdit, QMainWindow, QMessageBox, QProgressBar,
    QPushButton, QSpinBox, QStatusBar, QTableWidget, QTableWidgetItem,
    QTabWidget, QTextEdit, QVBoxLayout, QWidget,
)

from watchdog.observers import Observer

from batch_sorter import DocumentSorter
from handlers.AudioClassifier import AudioClassifier
from handlers.db_handler import DBHandler, DBHandlerError
from handlers.image_handler import ImageHandler
from handlers.ImageProcessor import ImageProcessor
from handlers.text_handler import TextHandler
from handlers.TextClassifier import TextClassifier
from handlers.VideoClassifier import VideoClassifier
from watchdog_service import DoCAEventHandler

logger = logging.getLogger("doca.gui")


# ---------------------------------------------------------------------- #
# Shared handler pool
# ---------------------------------------------------------------------- #
class HandlerPool:
    """Lazy, process-wide pool of ML handlers.

    Each handler is constructed the first time it is requested and reused
    for every subsequent call. This avoids paying the NLTK / Gensim /
    OpenCV / Keras startup cost every time the user runs a batch sort or
    toggles the watchdog.
    """

    def __init__(self) -> None:
        self._text: Optional[TextHandler] = None
        self._image: Optional[ImageHandler] = None
        self._audio: Optional[AudioClassifier] = None
        self._video: Optional[VideoClassifier] = None
        self._db: Optional[DBHandler] = None
        # Watchdog-specific handler types (TextClassifier, ImageProcessor)
        self._text_classifier: Optional[TextClassifier] = None
        self._image_processor: Optional[ImageProcessor] = None

    # Each property logs once, the first time it is hit.
    @property
    def text(self) -> TextHandler:
        if self._text is None:
            logger.info("Loading TextHandler (NLTK + Doc2Vec) ...")
            self._text = TextHandler()
        return self._text

    @property
    def image(self) -> ImageHandler:
        if self._image is None:
            logger.info("Loading ImageHandler (OCR + CV) ...")
            self._image = ImageHandler()
        return self._image

    @property
    def audio(self) -> AudioClassifier:
        if self._audio is None:
            logger.info("Loading AudioClassifier (inaSpeechSegmenter) ...")
            self._audio = AudioClassifier()
        return self._audio

    @property
    def video(self) -> VideoClassifier:
        if self._video is None:
            logger.info("Loading VideoClassifier (OpenCV + SSIM) ...")
            self._video = VideoClassifier()
        return self._video

    @property
    def db(self) -> DBHandler:
        if self._db is None:
            logger.info("Connecting to CouchDB ...")
            self._db = DBHandler()
        return self._db

    @property
    def text_classifier(self) -> TextClassifier:
        """TextClassifier for the watchdog (different class from TextHandler)."""
        if self._text_classifier is None:
            logger.info("Loading TextClassifier (NLTK + Doc2Vec) ...")
            self._text_classifier = TextClassifier()
        return self._text_classifier

    @property
    def image_processor(self) -> ImageProcessor:
        """ImageProcessor for the watchdog (different class from ImageHandler)."""
        if self._image_processor is None:
            logger.info("Loading ImageProcessor (OCR + CV + PDF) ...")
            self._image_processor = ImageProcessor()
        return self._image_processor

    def build_sorter(self) -> DocumentSorter:
        """Return a :class:`DocumentSorter` that reuses the pooled handlers."""
        return DocumentSorter(
            db_handler=self.db,
            text_handler=self.text,
            image_handler=self.image,
            audio_handler=self.audio,
            video_handler=self.video,
        )

    def build_event_handler(self, output_dir: str) -> DoCAEventHandler:
        """Return a :class:`DoCAEventHandler` that reuses the pooled handlers.

        Note: DoCAEventHandler expects TextClassifier and ImageProcessor
        (not TextHandler / ImageHandler used by the batch sorter).
        """
        return DoCAEventHandler(
            db_handler=self.db,
            text_handler=self.text_classifier,
            image_handler=self.image_processor,
            video_handler=self.video,
            audio_handler=self.audio,
            output_dir=output_dir,
        )


# ---------------------------------------------------------------------- #
# Logging plumbing
# ---------------------------------------------------------------------- #
class QtLogHandler(logging.Handler, QObject):
    """Logging handler that emits a Qt signal for each formatted record."""

    log_emitted = pyqtSignal(str)

    def __init__(self) -> None:
        logging.Handler.__init__(self)
        QObject.__init__(self)
        self.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D401
        try:
            self.log_emitted.emit(self.format(record))
        except Exception:
            self.handleError(record)


# ---------------------------------------------------------------------- #
# Small UI helpers
# ---------------------------------------------------------------------- #
def _hline() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setFrameShadow(QFrame.Shadow.Sunken)
    return line


class DirectoryPicker(QWidget):
    """A label + line-edit + Browse button for choosing a directory."""

    def __init__(self, label: str, default: str = "") -> None:
        super().__init__()
        self.edit = QLineEdit(default)
        self.edit.setPlaceholderText(label)
        self.button = QPushButton("Browse…")
        self.button.clicked.connect(self._on_browse)

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        label_w = QLabel(label)
        label_w.setMinimumWidth(170)
        row.addWidget(label_w, 0)
        row.addWidget(self.edit, 1)
        row.addWidget(self.button, 0)

    def path(self) -> str:
        return self.edit.text().strip()

    def set_path(self, value: str) -> None:
        self.edit.setText(value)

    def _on_browse(self) -> None:
        start = self.path() or os.getcwd()
        chosen = QFileDialog.getExistingDirectory(self, "Select directory", start)
        if chosen:
            self.edit.setText(chosen)


# ---------------------------------------------------------------------- #
# Workers
# ---------------------------------------------------------------------- #
class BatchSortWorker(QObject):
    """Runs :meth:`DocumentSorter.sort_directory` on a worker thread."""

    progress = pyqtSignal(int, int, str, object)  # index, total, path, category
    finished = pyqtSignal(dict)                    # result dict
    failed = pyqtSignal(str)                       # error message

    def __init__(
        self,
        pool: HandlerPool,
        input_dir: str,
        output_dir: str,
        threshold: int,
        recursive: bool = False,
    ) -> None:
        super().__init__()
        self._pool = pool
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.threshold = threshold
        self.recursive = recursive

    @pyqtSlot()
    def run(self) -> None:
        try:
            sorter = self._pool.build_sorter()
            result = sorter.sort_directory(
                self.input_dir,
                self.output_dir,
                threshold=self.threshold,
                recursive=self.recursive,
                progress_cb=self._on_progress,
            )
            self.finished.emit(result)
        except Exception as exc:
            logger.exception("Batch sort failed")
            self.failed.emit(str(exc))

    def _on_progress(
        self, index: int, total: int, path: str, category: Optional[str]
    ) -> None:
        # Emitted from the worker thread; Qt queues delivery to GUI-thread slots.
        self.progress.emit(index, total, path, category)


class WatchdogWorker(QObject):
    """Runs a watchdog Observer on a worker thread."""

    started_ok = pyqtSignal(str)    # watch path
    stopped = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(
        self,
        pool: HandlerPool,
        watch_dir: str,
        output_dir: str,
        recursive: bool = False,
    ) -> None:
        super().__init__()
        self._pool = pool
        self.watch_dir = watch_dir
        self.output_dir = output_dir
        self.recursive = recursive
        self._observer: Optional[Observer] = None

    @pyqtSlot()
    def start(self) -> None:
        try:
            os.makedirs(self.watch_dir, exist_ok=True)
            os.makedirs(self.output_dir, exist_ok=True)

            event_handler = self._pool.build_event_handler(self.output_dir)

            self._observer = Observer()
            self._observer.schedule(
                event_handler, self.watch_dir, recursive=self.recursive
            )
            self._observer.start()
            self.started_ok.emit(self.watch_dir)
        except Exception as exc:
            logger.exception("Failed to start watchdog")
            self.failed.emit(str(exc))

    @pyqtSlot()
    def stop(self) -> None:
        try:
            if self._observer is not None:
                self._observer.stop()
                self._observer.join(timeout=10)
                self._observer = None
            self.stopped.emit()
        except Exception as exc:
            logger.exception("Failed to stop watchdog cleanly")
            self.failed.emit(str(exc))


class DBFetchWorker(QObject):
    """Fetches all records from CouchDB on a worker thread."""

    fetched = pyqtSignal(list)   # list of dicts
    failed = pyqtSignal(str)

    def __init__(self, pool: HandlerPool) -> None:
        super().__init__()
        self._pool = pool

    @pyqtSlot()
    def run(self) -> None:
        try:
            db = self._pool.db
            rows: list[dict[str, Any]] = []
            # couchdb.Database iterates over document IDs.
            for doc_id in db.db:
                # Skip CouchDB design documents.
                if str(doc_id).startswith("_design/"):
                    continue
                doc = db.db.get(doc_id)
                if doc is None:
                    continue
                rows.append(dict(doc))
            self.fetched.emit(rows)
        except DBHandlerError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:
            logger.exception("Failed to fetch DB records")
            self.failed.emit(str(exc))


# ---------------------------------------------------------------------- #
# Tab 1: Batch Sorter
# ---------------------------------------------------------------------- #
class BatchSorterTab(QWidget):
    def __init__(self, pool: HandlerPool) -> None:
        super().__init__()
        self._pool = pool
        self._thread: Optional[QThread] = None
        self._worker: Optional[BatchSortWorker] = None

        self.input_picker = DirectoryPicker(
            "Input Directory", os.path.abspath("./monitored_folder")
        )
        self.output_picker = DirectoryPicker(
            "Output Directory", os.path.abspath("./organized_output")
        )

        self.threshold = QSpinBox()
        self.threshold.setRange(1, 1000)
        self.threshold.setValue(3)

        self.recursive = QCheckBox("Recurse into subdirectories")

        self.start_btn = QPushButton("Start Batch Sort")
        self.start_btn.setMinimumHeight(40)
        self.start_btn.clicked.connect(self._on_start)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)

        self.status = QLabel("Idle.")
        self.status.setWordWrap(True)

        self.summary = QTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setPlaceholderText(
            "Per-category tallies will appear here after the run completes."
        )

        form = QFormLayout()
        form.addRow("Threshold:", self.threshold)
        form.addRow("", self.recursive)

        layout = QVBoxLayout(self)
        layout.addWidget(self.input_picker)
        layout.addWidget(self.output_picker)
        layout.addLayout(form)
        layout.addWidget(self.start_btn)
        layout.addWidget(self.progress)
        layout.addWidget(self.status)
        layout.addWidget(_hline())
        layout.addWidget(QLabel("Run summary:"))
        layout.addWidget(self.summary, 1)

    # ------------------------------------------------------------------ #
    def _on_start(self) -> None:
        input_dir = self.input_picker.path()
        output_dir = self.output_picker.path()

        if not input_dir or not os.path.isdir(input_dir):
            QMessageBox.warning(self, "DoCA", "Please choose a valid input directory.")
            return
        if not output_dir:
            QMessageBox.warning(self, "DoCA", "Please choose an output directory.")
            return

        self.start_btn.setEnabled(False)
        self.progress.setRange(0, 0)  # indeterminate until first progress tick
        self.status.setText("Starting batch sort...")
        self.summary.clear()

        self._thread = QThread(self)
        self._worker = BatchSortWorker(
            pool=self._pool,
            input_dir=input_dir,
            output_dir=output_dir,
            threshold=self.threshold.value(),
            recursive=self.recursive.isChecked(),
        )
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)

        # Lifecycle cleanup
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.finished.connect(self._clear_worker)

        self._thread.start()

    @pyqtSlot(int, int, str, object)
    def _on_progress(self, index: int, total: int, path: str, category: object) -> None:
        if self.progress.maximum() != total:
            self.progress.setRange(0, max(total, 1))
        self.progress.setValue(index)
        cat = category if category else "skipped"
        self.status.setText(f"[{index}/{total}] {os.path.basename(path)} → {cat}")

    @pyqtSlot(dict)
    def _on_finished(self, result: dict) -> None:
        self.start_btn.setEnabled(True)
        self.progress.setRange(0, max(self.progress.maximum(), 1))
        self.progress.setValue(self.progress.maximum())
        tallies = result.get("tallies", {})
        moved = result.get("moved", [])
        skipped = result.get("skipped", [])
        self.status.setText(
            f"Done. Processed {sum(tallies.values())}, moved {len(moved)}, "
            f"skipped {len(skipped)}."
        )
        lines = ["Tallies:"]
        for k, v in sorted(tallies.items()):
            lines.append(f"  {k}: {v}")
        lines.append("")
        lines.append(f"Threshold: {result.get('threshold')}")
        lines.append(f"Output:    {result.get('output_dir')}")
        if skipped:
            lines.append(f"\nSkipped {len(skipped)} file(s):")
            for item in skipped[:50]:
                lines.append(f"  - {item['path']} ({item.get('reason')})")
            if len(skipped) > 50:
                lines.append(f"  ... and {len(skipped) - 50} more")
        self.summary.setPlainText("\n".join(lines))

    @pyqtSlot(str)
    def _on_failed(self, message: str) -> None:
        self.start_btn.setEnabled(True)
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.status.setText(f"Failed: {message}")
        QMessageBox.critical(self, "Batch sort failed", message)

    @pyqtSlot()
    def _clear_worker(self) -> None:
        self._worker = None
        self._thread = None


# ---------------------------------------------------------------------- #
# Tab 2: Live Watchdog
# ---------------------------------------------------------------------- #
class WatchdogTab(QWidget):
    # Thread-safe way to ask the worker to start/stop.
    _request_start = pyqtSignal()
    _request_stop = pyqtSignal()

    def __init__(self, pool: HandlerPool, log_handler: QtLogHandler) -> None:
        super().__init__()
        self._pool = pool
        self._log_handler = log_handler
        self._thread: Optional[QThread] = None
        self._worker: Optional[WatchdogWorker] = None

        self.watch_picker = DirectoryPicker(
            "Directory to Monitor", os.path.abspath("./monitored_folder")
        )
        self.output_picker = DirectoryPicker(
            "Organized Output", os.path.abspath("./organized_output")
        )
        self.recursive = QCheckBox("Recurse into subdirectories")

        self.start_btn = QPushButton("Start Watchdog")
        self.stop_btn = QPushButton("Stop Watchdog")
        self.stop_btn.setEnabled(False)
        self.start_btn.clicked.connect(self._on_start_clicked)
        self.stop_btn.clicked.connect(self._on_stop_clicked)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.stop_btn)
        btn_row.addStretch(1)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self.log_view.setStyleSheet(
            "QTextEdit { font-family: Menlo, Consolas, monospace; "
            "background: #111; color: #ddd; }"
        )

        clear_btn = QPushButton("Clear Log")
        clear_btn.clicked.connect(self.log_view.clear)
        log_header = QHBoxLayout()
        log_header.addWidget(QLabel("Live log:"))
        log_header.addStretch(1)
        log_header.addWidget(clear_btn)

        layout = QVBoxLayout(self)
        layout.addWidget(self.watch_picker)
        layout.addWidget(self.output_picker)
        layout.addWidget(self.recursive)
        layout.addLayout(btn_row)
        layout.addLayout(log_header)
        layout.addWidget(self.log_view, 1)

        # Forward every log record the handler emits to the view.
        self._log_handler.log_emitted.connect(self._append_log)

    # ------------------------------------------------------------------ #
    @pyqtSlot(str)
    def _append_log(self, line: str) -> None:
        self.log_view.append(line)
        cursor = self.log_view.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.log_view.setTextCursor(cursor)

    def _on_start_clicked(self) -> None:
        watch = self.watch_picker.path()
        out = self.output_picker.path()
        if not watch:
            QMessageBox.warning(self, "DoCA", "Please choose a directory to monitor.")
            return
        if not out:
            QMessageBox.warning(self, "DoCA", "Please choose an organized output directory.")
            return

        self.start_btn.setEnabled(False)
        self._append_log(f"Starting watchdog on {watch} -> {out}")

        self._thread = QThread(self)
        self._worker = WatchdogWorker(
            self._pool, watch, out, recursive=self.recursive.isChecked()
        )
        self._worker.moveToThread(self._thread)

        self._request_start.connect(self._worker.start)
        self._request_stop.connect(self._worker.stop)

        self._worker.started_ok.connect(self._on_started_ok)
        self._worker.stopped.connect(self._on_stopped)
        self._worker.failed.connect(self._on_failed)

        self._thread.start()
        self._request_start.emit()

    def _on_stop_clicked(self) -> None:
        if self._worker is None:
            return
        self.stop_btn.setEnabled(False)
        self._append_log("Stopping watchdog...")
        self._request_stop.emit()

    @pyqtSlot(str)
    def _on_started_ok(self, path: str) -> None:
        self.stop_btn.setEnabled(True)
        self._append_log(f"Watchdog running on {path}.")

    @pyqtSlot()
    def _on_stopped(self) -> None:
        self._append_log("Watchdog stopped.")
        self._teardown_thread()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    @pyqtSlot(str)
    def _on_failed(self, message: str) -> None:
        self._append_log(f"Watchdog error: {message}")
        self._teardown_thread()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        QMessageBox.critical(self, "Watchdog error", message)

    def _teardown_thread(self) -> None:
        if self._worker is not None:
            self._worker.deleteLater()
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(5000)
            self._thread.deleteLater()
        self._worker = None
        self._thread = None

    def shutdown(self) -> None:
        """Called when the main window is closing."""
        if self._worker is not None:
            self._request_stop.emit()
            if self._thread is not None:
                self._thread.quit()
                self._thread.wait(5000)


# ---------------------------------------------------------------------- #
# Tab 3: Database Dashboard
# ---------------------------------------------------------------------- #
class DatabaseTab(QWidget):
    COLUMNS = ("File Name", "Category", "Extension", "Stored Path")

    def __init__(self, pool: HandlerPool) -> None:
        super().__init__()
        self._pool = pool
        self._thread: Optional[QThread] = None
        self._worker: Optional[DBFetchWorker] = None

        self.refresh_btn = QPushButton("Refresh Data")
        self.refresh_btn.setMinimumHeight(32)
        self.refresh_btn.clicked.connect(self._on_refresh)
        self.status = QLabel("Click 'Refresh Data' to load records from CouchDB.")
        self.status.setWordWrap(True)

        header_row = QHBoxLayout()
        header_row.addWidget(self.refresh_btn)
        header_row.addWidget(self.status, 1)

        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(True)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)

        layout = QVBoxLayout(self)
        layout.addLayout(header_row)
        layout.addWidget(self.table, 1)

    def _on_refresh(self) -> None:
        self.refresh_btn.setEnabled(False)
        self.status.setText("Loading…")

        self._thread = QThread(self)
        self._worker = DBFetchWorker(self._pool)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.fetched.connect(self._on_fetched)
        self._worker.failed.connect(self._on_failed)

        # Lifecycle cleanup
        self._worker.fetched.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.finished.connect(self._clear_worker)

        self._thread.start()

    @pyqtSlot(list)
    def _on_fetched(self, rows: list) -> None:
        self.refresh_btn.setEnabled(True)

        # Defensive filter (worker already drops these, but be explicit).
        rows = [r for r in rows if not str(r.get("_id", "")).startswith("_design/")]

        self.table.setRowCount(len(rows))
        for row_idx, doc in enumerate(rows):
            file_name = doc.get("file_name") or os.path.basename(
                doc.get("stored_path") or doc.get("path", "")
            )
            category = (
                doc.get("category")
                or doc.get("category_folder")
                or doc.get("file_type", "")
            )
            extension = doc.get("extension") or os.path.splitext(
                doc.get("stored_path") or doc.get("path", "")
            )[1]
            stored = doc.get("stored_path") or doc.get("path", "")
            values = (file_name, str(category), extension, stored)
            for col_idx, value in enumerate(values):
                item = QTableWidgetItem(str(value) if value is not None else "")
                item.setToolTip(str(value))
                self.table.setItem(row_idx, col_idx, item)

        self.status.setText(f"Loaded {len(rows)} record(s).")

    @pyqtSlot(str)
    def _on_failed(self, message: str) -> None:
        self.refresh_btn.setEnabled(True)
        self.status.setText(f"Failed: {message}")
        QMessageBox.critical(self, "Database error", message)

    @pyqtSlot()
    def _clear_worker(self) -> None:
        self._worker = None
        self._thread = None


# ---------------------------------------------------------------------- #
# Styling
# ---------------------------------------------------------------------- #
_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #f4f6fa;
    color: #1f2330;
    font-size: 10.5pt;
}
QTabWidget::pane {
    border: 1px solid #c8cdd8;
    border-radius: 6px;
    background: #ffffff;
    top: -1px;
}
QTabBar::tab {
    background: #e4e8f0;
    color: #333a4a;
    padding: 8px 20px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    margin-right: 2px;
}
QTabBar::tab:selected {
    background: #ffffff;
    color: #1447e6;
    font-weight: 600;
}
QPushButton {
    background-color: #2b5bff;
    color: white;
    border: none;
    padding: 8px 16px;
    border-radius: 6px;
    font-weight: 500;
}
QPushButton:hover  { background-color: #1d46d6; }
QPushButton:pressed{ background-color: #1638a8; }
QPushButton:disabled {
    background-color: #b9c0d1;
    color: #f0f0f0;
}
QLineEdit, QSpinBox, QTextEdit {
    background: #ffffff;
    border: 1px solid #c8cdd8;
    border-radius: 4px;
    padding: 6px;
    selection-background-color: #2b5bff;
}
QProgressBar {
    border: 1px solid #c8cdd8;
    border-radius: 6px;
    background: #ffffff;
    text-align: center;
    min-height: 18px;
}
QProgressBar::chunk {
    background-color: #2b5bff;
    border-radius: 5px;
}
QTableWidget {
    background-color: #ffffff;
    alternate-background-color: #f8fafc;
    color: #0f172a;
    gridline-color: #e2e8f0;
    border: 1px solid #c8cdd8;
    border-radius: 6px;
}
QTableWidget::item {
    color: #0f172a;
}
QTableWidget::item:selected {
    background-color: #2563eb;
    color: #ffffff;
}
QHeaderView::section {
    background: #e4e8f0;
    padding: 6px;
    border: none;
    font-weight: 600;
}
QStatusBar {
    background: #e4e8f0;
    color: #333a4a;
}
"""


# ---------------------------------------------------------------------- #
# Main window
# ---------------------------------------------------------------------- #
class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("DoCA — Document Classification and Analysis")
        self.resize(1100, 750)

        # 1. One shared handler pool for the entire application.
        self.pool = HandlerPool()

        # 2. Install the Qt log handler on the root logger so every module
        #    (batch_sorter, watchdog_service, handlers.*) feeds the live log.
        self.log_handler = QtLogHandler()
        self.log_handler.setLevel(logging.INFO)
        logging.getLogger().addHandler(self.log_handler)

        # 3. Build the three tabs.
        self.batch_tab = BatchSorterTab(self.pool)
        self.watchdog_tab = WatchdogTab(self.pool, self.log_handler)
        self.db_tab = DatabaseTab(self.pool)

        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        tabs.addTab(self.batch_tab, "Batch Sorter")
        tabs.addTab(self.watchdog_tab, "Live Watchdog")
        tabs.addTab(self.db_tab, "Database")
        self.setCentralWidget(tabs)

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Ready.")

        quit_action = QAction("Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        self.menuBar().addMenu("&File").addAction(quit_action)

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt API)
        try:
            self.watchdog_tab.shutdown()
        finally:
            # Detach the log handler so stray shutdown logs don't hit a dead view.
            logging.getLogger().removeHandler(self.log_handler)
            super().closeEvent(event)


# ---------------------------------------------------------------------- #
# Entry point
# ---------------------------------------------------------------------- #
def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    app = QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName("DoCA")
    app.setStyle("Fusion")
    app.setStyleSheet(_STYLESHEET)

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
