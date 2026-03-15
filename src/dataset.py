# Qui dentro viene messa la classe Dataset
# Deve contenere i metodi
# __init__
# __len__
# __getitem__
# Per fare Data Augmentation è meglio definirle Qui
# Il vantaggio è che possiamo importare dataset senza preoccuparci
# di come vengono letti i file CSV, le immagini o i file audio

import torch
import torchaudio
import pandas as pd
import numpy as np
import random
from torch.utils.data import Dataset
from pathlib import Path

class MusicNetPianoDataset(Dataset):
    def __init__(self, 
                 csv_file = "data/solo_piano.csv", 
                 data_dir="data/raw", 
                 split='train', 
                 chunk_duration=5.0, 
                 sample_rate=22050
                 ):
        # Reads the .csv and filters only for 'train' and 'test'
        df = pd.read_csv(csv_file)
        self.data = df[df['split'] == split].reset_index(drop=True)

        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.chunk_samples = int(chunk_duration * sample_rate)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        track_id = str[row['id']]

        wav_path = self.data_dir / "wav" / f"{track_id}.wav"
        label_path = self.data_dir / "labels" / f"{track_id}.npy"

        # obtains audio info
        info = torchaudio.info(wav_path)
        total_samples = info.num_frames
        orig_sr = info.sample_rate

        # choose a random point for extracting 5 seconds
        if total_samples > self.chunk_samples:
            start_frame = random.randint(0, total_samples - self.chunk_samples)
        else:
            start_frame = 0

        # load the 5 second frame
        waveform, sr = torchaudio.load(wav_path, frame_offset=start_frame, num_frames=self.chunk_samples)

        # Mono conversion
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # if the sampling rate changes, this re-samples at 22.05 kHz
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        # load labels
        # for the labels we need to synchronize the label to the 5 second interval that we have extraced
        # Because if I extract a 5 second segment, I must load the label( sheet music ) of these 5 seconds
        hop_length = 512
        start_cqt_frame = start_frame // hop_length
        num_cqt_frames = self.chunk_samples // hop_length
        
        labels = np.load(label_path) 

        chunk_labels = labels[
            start_cqt_frame : start_cqt_frame + num_cqt_frames, :
        ]
        chunk_labels = torch.tensor(chunk_labels, dtype=torch.float32)

        return {
            "waveform": waveform,
            "labels": labels, 
            "id": track_id
        }

'''

# --- TEST ----
if __name__ == "__main__":
    #
    # test dataset on a block
    # insert here

    dataset = MusicNetPianoDataset(split="train")
    print(f"Tracce di training trovate: {len(dataset)}")
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Shape dell'audio: {sample['waveform'].shape}") 

'''