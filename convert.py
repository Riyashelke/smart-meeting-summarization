import os
from pathlib import Path

from moviepy import AudioFileClip

from pipeline_config import MP3_DIR, RAW_AMI_DIR


def convert_rm_to_mp3(root_folder: Path = RAW_AMI_DIR, output_folder: Path = MP3_DIR) -> list[Path]:
    """
    Convert all .rm files under root_folder to .mp3 in output_folder.
    Returns list of written mp3 paths.
    """
    output_folder.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for subdir, _, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(".rm"):
                rm_path = Path(subdir) / file
                base_name = rm_path.stem
                mp3_path = output_folder / f"{base_name}.mp3"

                print(f"\n🔄 Converting: {rm_path}")
                print(f"➡ Saving to: {mp3_path}")

                try:
                    audio = AudioFileClip(str(rm_path))
                    audio.write_audiofile(str(mp3_path))
                    audio.close()
                    written.append(mp3_path)
                    print("✅ Done")
                except Exception as e:  # noqa: BLE001
                    print("❌ Error converting:", rm_path)
                    print("   Reason:", e)

    print(f"\n🎉 All conversions saved in: {output_folder}")
    return written


if __name__ == "__main__":
    convert_rm_to_mp3()