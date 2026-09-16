"""
Utility script to convert all dataset WAV files to high-fidelity MP3 files.
Reduces storage footprint by >90% and enables interactive web/markdown audio playback.
"""

# Import Modules
from pathlib import Path
import subprocess
import soundfile as sf

def main() -> None:
    output_dir = Path("data/output")
    wav_files = sorted(output_dir.rglob("*.wav"))

    print(f"Found {len(wav_files)} WAV files to convert to MP3...")
    converted = 0

    for wav_path in wav_files:
        mp3_path = wav_path.with_suffix(".mp3")
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(wav_path),
            "-codec:a",
            "libmp3lame",
            "-b:a",
            "192k",
            str(mp3_path),
        ]
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        if res.returncode == 0 and mp3_path.is_file():
            # Verify readability
            try:
                _ = sf.info(str(mp3_path))
                wav_path.unlink()
                converted += 1
            except Exception as e:
                print(f"Failed verifying {mp3_path}: {e}")
        else:
            print(f"Failed converting {wav_path}")

    print(f"Successfully converted {converted} files to MP3 and removed original WAVs.")

if __name__ == "__main__":
    main()
