import kagglehub
import pandas as pd
from pathlib import Path
import os

def main():
    print("Download MusicNet da Kaggle...")

    # -------- entire dataset download ----------
    path = kagglehub.dataset_download("imsparsh/musicnet-dataset")
    print(f"Dateset created in: {path}")

    # -------- kaggle dataset paths variables ----------
    root = Path(path) / "musicnet" / "musicnet"
    meta_path = Path(path) / "musicnet_metadata.csv"
    
    # reading the original csv
    meta = pd.read_csv(meta_path)
    # define an empty list, it will contains all the correct and verified data
    list = [] 

    # for each row of the original csv : 
    # - extract track id and instrument
    # - verify if it is in train_data or test_data
    # - adds a new element to the list

    for _, row in meta.iterrows():
        track_id = str(row["id"])
        ensemble = row["ensemble"]
        
        # train or test ? 
        if (root / "train_data" / f"{track_id}.wav").exists():
            split = "train"
        elif (root / "test_data" / f"{track_id}.wav").exists():
            split = "test"
        else:
            continue # if the file does not exist in both the folders, skip

        list.append({
            "id": track_id,
            "split": split,
            "ensemble": ensemble,
            "wav_path": str(root / f"{split}_data" / f"{track_id}.wav"),
            "label_path": str(root / f"{split}_labels" / f"{track_id}.npy"),
            "midi_path": str(Path(path) / "musicnet_midis" / "musicnet_midis" / f"{track_id}.mid")
        })

    # tranfosrms the list in a Data Frame , a table
    df = pd.DataFrame(list)
    
    # Makes the 'data' folder
    os.makedirs("data", exist_ok=True)

    # Filters the DataFrame and selects only the files labeled 'Solo Piano'
    solo_piano = df[df["ensemble"] == "Solo Piano"]

    # Creates a new .csv file with 'Solo Piano'
    solo_piano.to_csv("data/solo_piano.csv", index=False)
    print(f"Saved data/solo_piano.csv with {len(solo_piano)} tracks.")


# This prevents the script to be executed in a wrong way
# Apparently it is a best practice in Python to insert this check
if __name__ == "__main__":
    main()