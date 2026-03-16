# ---------- DESCRIPTION ---------------
# This script reads the .csv and copies only the 'solo piano' files
# in the local folder 'data/raw/'.

import pandas as pd
import shutil
import os
from pathlib import Path

def main():
    # csv path defined in download_and_filter.py script
    csv_path = "data/solo_piano.csv"
    # check if the path doesn't exists
    if not os.path.exists(csv_path):
        print("Error: You must execute download_and_filter.py")
        return

    # reads the csv containing 'Solo piano'
    df = pd.read_csv(csv_path)
    
    # define a variable containing the path of the 'data' folder
    raw_dir = Path("data/raw")

    # searches he subdirectories 'wav', 'midi', 'labels'
    for subdir in ["wav", "midi", "labels"]:
        # makes intermidieate folders ?
        (raw_dir / subdir).mkdir(parents=True, exist_ok=True)

    print("Copyng {len(df)} files 'Solo Piano' in data/raw/ ...")
    
    # For each row of the csv table :
    # - Takes the track id
    # - Copies the file .wav in data/raw/wav/
    # - Copies the file .midi in data/raw/midi/
    # - Copies the file .npy in data/raw/labels/
    for _, row in df.iterrows():
        track_id = str(row["id"])

        # COPY MIDI
        # retrieve the path from csv
        src_midi = row["midi_path"]
        if os.path.exists(src_midi):
            shutil.copy(src_midi, raw_dir / "midi" / f"{track_id}.mid")
        else:
            print(f"Warning: MIDI not found for {track_id} at {src_midi}")

        # COPY WAV
        src_wav = row["wav_path"]
        if os.path.exists(src_wav):
            shutil.copy(src_wav, raw_dir / "wav" / f"{track_id}.wav")
        else:
            print(f"Warning: WAV not found for {track_id}")
        
        # COPY LABELS
        src_label = row["label_path"]
        if os.path.exists(src_label):
            shutil.copy(src_label, raw_dir / "labels" /"labels" f"{track_id}.csv")
        else:
            print(f"Warning: Label not found for {track_id}")

    print(f"Process completed. Check {raw_dir} for files")

# This prevents the script to be executed in a wrong way
# Apparently it is a best practice in Python to insert this check
if __name__ == "__main__":
    main()