
import pandas as pd
import shutil
import os
from pathlib import Path

def main():
    csv_path = "data/solo_piano.csv"
    if not os.path.exists(csv_path):
        print("Error: You must execute download_and_filter.py")
        return

    df = pd.read_csv(csv_path)
    
    raw_dir = Path("data/raw")

    for subdir in ["wav", "midi", "labels"]:
        (raw_dir / subdir).mkdir(parents=True, exist_ok=True)

    print(f"Coping {len(df)} files 'Solo Piano' in data/raw/ ...")
    
    for _, row in df.iterrows():
        track_id = str(row["id"])

        src_midi = row["midi_path"]
        if os.path.exists(src_midi):
            shutil.copy(src_midi, raw_dir / "midi" / f"{track_id}.mid")
        else:
            print(f"Warning: MIDI not found for {track_id} at {src_midi}")

        src_wav = row["wav_path"]
        if os.path.exists(src_wav):
            shutil.copy(src_wav, raw_dir / "wav" / f"{track_id}.wav")
        else:
            print(f"Warning: WAV not found for {track_id}")
        
        src_label = row["label_path"]
        if os.path.exists(src_label):
            shutil.copy(src_label, raw_dir / "labels" /"labels" f"{track_id}.csv")
        else:
            print(f"Warning: Label not found for {track_id}")

    print(f"Process completed. Check {raw_dir} for files")

if __name__ == "__main__":
    main()