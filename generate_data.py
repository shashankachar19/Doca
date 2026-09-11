import os
import random
import numpy as np
import scipy.io.wavfile as wav
from pydub import AudioSegment
import cv2
from docx import Document
from fpdf import FPDF

OUT_DIR = "test_data"
NUM_FILES_PER_TYPE = 20

os.makedirs(OUT_DIR, exist_ok=True)

def generate_pdf(path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Lorem ipsum dolor sit amet", ln=True, align='L')
    pdf.output(path)

def generate_docx(path):
    doc = Document()
    doc.add_paragraph("Lorem ipsum dolor sit amet")
    doc.save(path)

def synthesize_audio(path, is_speech=False):
    # inaSpeechSegmenter uses a CNN. To trick it into speech vs music,
    # we can try to generate a specific frequency sine wave.
    # Speech generally has formants, but we'll try something simple.
    sample_rate = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    if is_speech:
        # Simulate speech-like frequencies (100-300Hz with some variation)
        freq = 200 + 50 * np.sin(2 * np.pi * 5 * t)
        signal = 0.5 * np.sin(2 * np.pi * freq * t)
    else:
        # Simulate music-like scale (440Hz A4)
        signal = 0.5 * np.sin(2 * np.pi * 440 * t)

    signal = np.int16(signal * 32767)
    tmp_wav = path.replace(".mp3", ".wav")
    wav.write(tmp_wav, sample_rate, signal)

    # Convert to MP3
    audio = AudioSegment.from_wav(tmp_wav)
    audio.export(path, format="mp3")
    os.remove(tmp_wav)

def generate_video(path, is_security=False):
    width, height = 640, 480
    fps = 24
    duration = 2
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))

    x, y = 320, 240
    for i in range(fps * duration):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        if not is_security:
            # Moving element for general video
            x = (x + 5) % width
            y = (y + 3) % height
            cv2.circle(frame, (x, y), 50, (0, 255, 0), -1)
        else:
            # Static frame with minor noise for security footage
            cv2.putText(frame, "CAM 01", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            noise = np.random.randint(0, 10, (height, width, 3), dtype=np.uint8)
            frame = cv2.add(frame, noise)

        out.write(frame)
    out.release()

def main():
    print("Generating PDFs and DOCX...")
    for i in range(NUM_FILES_PER_TYPE):
        generate_pdf(os.path.join(OUT_DIR, f"doc_{i:02d}.pdf"))
        generate_docx(os.path.join(OUT_DIR, f"doc_{i:02d}.docx"))

    print("Generating MP3s...")
    for i in range(NUM_FILES_PER_TYPE):
        # 10 speech, 10 music
        is_speech = i < (NUM_FILES_PER_TYPE // 2)
        synthesize_audio(os.path.join(OUT_DIR, f"audio_{i:02d}.mp3"), is_speech)

    print("Generating MP4s...")
    for i in range(NUM_FILES_PER_TYPE):
        # 10 security, 10 general
        is_security = i < (NUM_FILES_PER_TYPE // 2)
        generate_video(os.path.join(OUT_DIR, f"video_{i:02d}.mp4"), is_security)

    print("Copying files to monitored_folder...")
    os.makedirs("monitored_folder", exist_ok=True)
    os.system(f"cp -r {OUT_DIR}/* monitored_folder/")

    print("Data generation complete.")

if __name__ == "__main__":
    main()
