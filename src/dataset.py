# Qui dentro viene messa la classe Dataset
# Deve contenere i metodi
# __init__
# __len__
# __getitem__
# Per fare Data Augmentation è meglio definirle Qui
# Il vantaggio è che possiamo importare dataset senza preoccuparci
# di come vengono letti i file CSV, le immagini o i file audio

import torch
import pandas as pd
from torch.utils.data import Dataset
import torchaudio
from pathlib import Path
import numpy as np

class MusicNetPianoDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        csv_file : percorso al CSV filtrato Solo Piano
        root_dir : cartella root che contiene train_data, test_data, train_labels, test_labels
        transform : eventuali trasformazioni da applicare all'audio
        """
        self.data = pd.read_csv(csv_file)
        self.root_dir = Path(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Ottieni la riga corrispondente
        row = self.data.iloc[idx]

        # Costruisci i path assoluti
        wav_path = self.root_dir / row['wav_path']
        label_path = self.root_dir / row['label_path']
        midi_path = self.root_dir / row['midi_path']  # se vuoi caricare midi

        # Carica l'audio
        waveform, sr = torchaudio.load(wav_path)

        # Carica le label
        labels = np.load(label_path)

        # Applica eventuali trasformazioni sull'audio
        if self.transform:
            waveform = self.transform(waveform)

        # Puoi decidere come restituire i dati
        return {
            "waveform": waveform,   # Tensor (canali x samples)
            "sample_rate": sr,
            "labels": torch.tensor(labels, dtype=torch.float32),
            "id": row['id']
        }

import kagglehub
path = kagglehub.dataset_download("imsparsh/musicnet-dataset")

dataset = Dataset(
    csv_path="music_metadata_sp.csv",
    root_dir=f"{path}"
)

print(len(dataset))
audio_sample = dataset[0]["audio"]
