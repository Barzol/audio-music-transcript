import kagglehub
import pandas as pd
from pathlib import Path
import os

def main():
    print("Download MusicNet da Kaggle...")

    path = kagglehub.dataset_download("imsparsh/musicnet-dataset")
    print(f"Dataset created in: {path}")

    root = Path(path) / "musicnet" / "musicnet"
    meta_path = Path(path) / "musicnet_metadata.csv"
    midi_root = Path(path) / "musicnet_midis" / "musicnet_midis"
    
    meta = pd.read_csv(meta_path)
    lists = [] 


    print("Filtering tracks and locating MIDI files :")
    for _, row in meta.iterrows():
        track_id = str(row["id"])
        ensemble = row["ensemble"]
        
        if (root / "train_data" / f"{track_id}.wav").exists():
            split = "train"
        elif (root / "test_data" / f"{track_id}.wav").exists():
            split = "test"
        else:
            continue

        midi_files = list(midi_root.rglob(f"{track_id}*.mid*"))

        if not midi_files:
            print(f"Warning: MIDI not found for track {track_id}, skipping")
            continue

        midi_path = str(midi_files[0])

        lists.append({
            "id": track_id,
            "split": split,
            "ensemble": ensemble,
            "wav_path": str(root / f"{split}_data" / f"{track_id}.wav"),
            "label_path": str(root / f"{split}_labels" / f"{track_id}.csv"),
            "midi_path": midi_path
        })

    df = pd.DataFrame(lists)
    
    os.makedirs("data", exist_ok=True)

    solo_piano = df[df["ensemble"] == "Solo Piano"]

    solo_piano.to_csv("data/solo_piano.csv", index=False)
    print(f"Saved data/solo_piano.csv with {len(solo_piano)} tracks.")


if __name__ == "__main__":
    main()