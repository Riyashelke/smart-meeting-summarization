import os
from pathlib import Path
from typing import Iterable

import whisper

from pipeline_config import MP3_DIR, TRANSCRIPT_DIR, WHISPER_MODEL_SIZE


def transcribe_folder(input_folder: Path = MP3_DIR, output_folder: Path = TRANSCRIPT_DIR, model_size: str = WHISPER_MODEL_SIZE) -> list[Path]:
    """
    Transcribe all mp3 files in input_folder to text files in output_folder.
    Returns list of transcript paths.
    """
    model = whisper.load_model(model_size)
    output_folder.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for file in os.listdir(input_folder):
        if file.lower().endswith(".mp3"):
            audio_path = Path(input_folder) / file
            print(f"Transcribing: {file}")

            result = model.transcribe(str(audio_path))

            txt_path = output_folder / file.replace(".mp3", ".txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(result["text"])

            written.append(txt_path)
            print(f"Saved: {txt_path}")

    print("✅ Done! All files transcribed locally.")
    return written


def transcribe_files(files: Iterable[Path], output_folder: Path = TRANSCRIPT_DIR, model_size: str = WHISPER_MODEL_SIZE) -> list[Path]:
    """
    Transcribe a provided iterable of mp3 paths.
    """
    model = whisper.load_model(model_size)
    output_folder.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for audio_path in files:
        if audio_path.suffix.lower() != ".mp3":
            continue
        print(f"Transcribing: {audio_path.name}")
        result = model.transcribe(str(audio_path))
        txt_path = output_folder / f"{audio_path.stem}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(result["text"])
        written.append(txt_path)
        print(f"Saved: {txt_path}")

    print("✅ Done! Transcribed provided files.")
    return written


if __name__ == "__main__":
    transcribe_folder()