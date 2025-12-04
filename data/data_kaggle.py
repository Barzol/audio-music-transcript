import kagglehub

# Download latest version
path = kagglehub.dataset_download("imsparsh/musicnet-dataset")

print("Path to dataset files:", path)

# Genero un nuovo csv composto da id, split, ensemble, wav_path, label_path, midi_path
import pandas as pd
import os
from pathlib import Path

meta = pd.read_csv(f"{path}/musicnet_metadata.csv")
rows = []

for _, row in meta.iterrows():
    id = row["id"]
    ensemble = row["ensemble"]
    root = Path(path) / "musicnet" / "musicnet"


    train_wav = root / "train_data" / f"{id}.wav"
    test_wav  = root / "test_data" / f"{id}.wav"

    if train_wav.exists():
        split = "train"
        wav_path = Path(split + "_data") / f"{id}.wav"  # path relativo per CSV
        label_path = Path(split + "_labels") / f"{id}.npy"
    elif test_wav.exists():
        split = "test"
        wav_path = Path(split + "_data") / f"{id}.wav"
        label_path = Path(split + "_labels") / f"{id}.npy"
    else:
        print(f"ID {id} non trovato")



    # costruzione dei path
    wav_path = f"{split}_data/{id}.wav"
    label_path = f"{split}_labels/{id}.npy"
    midi_path = f"musicnet_midis/{id}.mid"

    rows.append({
        "id": id,
        "split": split,
        "ensemble": ensemble,
        "wav_path": wav_path,
        "label_path": label_path,
        "midi_path": midi_path
    })


df = pd.DataFrame(rows)

# Seleziono Piano Solo
solo_piano = df[df["ensemble"] == "Solo Piano"]
solo_piano.to_csv("data/solo_piano.csv", index=False)



