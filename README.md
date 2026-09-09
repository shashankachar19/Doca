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

## Installation & Setup

1. Clone the repository.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure CouchDB is installed and running locally.
4. Run the main dashboard:
   ```bash
   python main_dashboard.py
   ```

*(Note: The `majorproject synopsis ppt.pptx` file analysis will be incorporated once the file is uploaded to the repository.)*
