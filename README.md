# DoCA - Document Classification and Analysis

## Project Overview

DoCA is a Content-Based Automatic Classification System over digital documents. It provides a unified framework to analyze and classify various file types, including text documents, images, audio, and video files. 

This project is an implementation of the framework proposed in the IEEE Access paper: *"DoCA: A Content-Based Automatic Classification System Over Digital Documents"* by Süleyman Eken, Houssem Menhour, and Kübra Köksal.

## Comparison to the Base Paper

This project accurately reflects the architecture and methodologies described in the base paper:

- **Document/Text Analysis**: Utilizes `nltk` and `gensim` (Doc2Vec) for natural language processing and text classification, matching the paper's approach to text document processing.
- **Image Analysis**: Employs Optical Character Recognition (OCR) via `pytesseract` and template matching/computer vision using `opencv-python` and `scikit-image`.
- **Audio Analysis**: Integrates the `inaSpeechSegmenter` library, a CNN-based audio segmentation toolkit, precisely as described in the paper for speech/music and gender classification.
- **Video Analysis**: Uses OpenCV to process video frames and computes the Structural Similarity Index (SSIM) to classify videos (e.g., distinguishing security camera footage based on stationary backgrounds).
- **File and Folder Watchdog**: Implements background directory monitoring using the Python `watchdog` library to auto-classify newly added files and track historical changes.
- **Database Integration**: Adopts Apache CouchDB for document-oriented NoSQL storage, preserving file metadata and categorization results.
- **User Interface**: Provides a graphical user interface built with `PyQt6` to manage batch sorting, monitor live watchdog logs, and browse the CouchDB database.

## Current Project Status

The project is currently fully functional and covers the core requirements outlined in the original research:

1. **Batch Sorter**: Operational. Can recursively scan a directory and sort documents into their respective categories based on content analysis.
2. **Live Watchdog**: Operational. Capable of monitoring a target directory for file creation events and processing them in real-time.
3. **Database Dashboard**: Operational. Connects to CouchDB to display the processed records and their assigned categories.
4. **Handlers**: All ML/Analysis handlers (Text, Image, Audio, Video, DB) are implemented and pooled efficiently for performance.

## Platform Support

| Platform | Status | Notes |
|----------|--------|-------|
| **Windows** | ✅ Fully supported | Use `python-magic-bin` (automatic) |
| **Linux** | ✅ Fully supported | Full audio speech/music classification |
| **macOS** | ✅ Fully supported | Use `brew install libmagic tesseract ffmpeg` |

> **Note:** Audio speech/music classification via `inaSpeechSegmenter` is only available on **Linux**. On Windows and macOS, audio files are still catalogued and classified, but they are tagged as "Audio" instead of being split into "Audio_Speech" / "Audio_Music".

---

## Installation & Setup

### Prerequisites

| Tool | Purpose | Install Guide |
|------|---------|---------------|
| **Python 3.10–3.12** | Runtime | [python.org](https://www.python.org/downloads/) |
| **Tesseract OCR** | Image text extraction | See [platform instructions below](#tesseract-ocr) |
| **FFmpeg** | Audio/video processing | See [platform instructions below](#ffmpeg) |
| **CouchDB** | Metadata database | See [CouchDB setup below](#couchdb-setup) |

### Step 1: Clone & Create Virtual Environment

```bash
git clone https://github.com/your-username/Doca.git
cd Doca
```

**Windows (PowerShell):**
```powershell
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

### Step 2: Install Python Dependencies

```bash
pip install -r requirements.txt
```

**Linux only** — for full audio speech/music classification:
```bash
pip install inaSpeechSegmenter
```

### Step 3: Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` with your CouchDB credentials:

```dotenv
COUCHDB_URL=http://127.0.0.1:5984
COUCHDB_USER=admin
COUCHDB_PASSWORD=your_secure_password
COUCHDB_DB_NAME=doca_db
```

> ⚠️ **Security**: Never commit `.env` to version control. It is already in `.gitignore`.

### Step 4: Set Up CouchDB

#### Option A: Docker (Recommended — all platforms)

```bash
docker run -d \
  --name doca-couchdb \
  -e COUCHDB_USER=admin \
  -e COUCHDB_PASSWORD=your_secure_password \
  -p 5984:5984 \
  couchdb:3
```

Verify it's running:
```bash
curl http://localhost:5984/
```

#### Option B: Native Install

- **Windows**: Download from [couchdb.apache.org](https://couchdb.apache.org/#download)
- **Linux**: `sudo apt install couchdb` (Debian/Ubuntu)
- **macOS**: `brew install couchdb`

### Step 5: Run the Dashboard

```bash
python ui/main_dashboard.py
```

---

## System Dependencies

### Tesseract OCR

- **Windows**: Download from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki) → install → add to PATH
- **Linux**: `sudo apt install tesseract-ocr`
- **macOS**: `brew install tesseract`

### FFmpeg

- **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html) → extract → add `bin/` to PATH
- **Linux**: `sudo apt install ffmpeg`
- **macOS**: `brew install ffmpeg`

---

## Usage

### GUI Dashboard

The main application (`python ui/main_dashboard.py`) provides three tabs:

| Tab | Description |
|-----|-------------|
| **Batch Sorter** | Select an input folder → classify & sort all documents into category subfolders |
| **Live Watchdog** | Monitor a directory in real-time; auto-classify newly added files |
| **Database** | Browse all processed records stored in CouchDB |

### CLI Watchdog Service

```bash
python watchdog_service.py --path ./monitored_folder --output ./organized_output
```

### Batch Sort (CLI)

```python
from batch_sorter import DocumentSorter

sorter = DocumentSorter()
result = sorter.sort_directory("/path/to/input", "/path/to/output", threshold=3)
print(result["tallies"])
```

---

## Project Structure

```
Doca/
├── ui/
│   └── main_dashboard.py    # PyQt6 GUI with 3 tabs
├── handlers/
│   ├── AudioClassifier.py   # inaSpeechSegmenter (graceful fallback)
│   ├── ImageProcessor.py    # OCR + SIFT + PDF (watchdog)
│   ├── image_handler.py     # OCR + SIFT (batch sorter)
│   ├── TextClassifier.py    # NLP + Doc2Vec + clustering (watchdog)
│   ├── text_handler.py      # NLP + Doc2Vec (batch sorter)
│   ├── VideoClassifier.py   # SSIM-based video analysis
│   └── db_handler.py        # CouchDB CRUD wrapper
├── batch_sorter.py          # Content-based file classifier & router
├── watchdog_service.py      # Directory monitor daemon
├── generate_data.py         # Synthetic test data generator
├── wait_for_db.py           # CouchDB readiness check
├── requirements.txt         # Python dependencies
├── .env.example             # Environment variable template
└── README.md
```

*(Note: The `majorproject synopsis ppt.pptx` file analysis will be incorporated once the file is uploaded to the repository.)*
