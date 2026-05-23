# dataset.py  –  Fase 2: Multi-Output
#
# Rispetto alla Fase 1, __getitem__ restituisce tre matrici di label:
#   labels  : piano roll binario  (nota attiva per ogni frame)  → shape (T, 84)
#   onsets  : onset roll binario  (1 solo al primo frame di ogni nota) → shape (T, 84)
#   offsets : offset roll binario (1 solo all'ultimo frame di ogni nota) → shape (T, 84)
#
# Tutto il resto (caricamento audio, CQT alignment, padding) è invariato.

import torch
import torchaudio
import pandas as pd
import numpy as np
import random
from torch.utils.data import Dataset
from pathlib import Path
import soundfile as sf


class MusicNetPianoDataset(Dataset):

    '''
    csv_file        : path al csv
    data_dir        : cartella radice dei dati
    split           : 'train' o 'test'
    chunk_duration  : durata in secondi di ogni chunk audio
    sample_rate     : frequenza di campionamento
    '''

    def __init__(self,
                 csv_file="data/solo_piano.csv",
                 data_dir="data/raw",
                 split='train',
                 chunk_duration=5.0,
                 sample_rate=22050):

        project_root = Path(__file__).parent.parent
        csv_path = project_root / csv_file
        self.data_dir = project_root / data_dir

        df = pd.read_csv(csv_path)
        self.data = df[df['split'] == split].reset_index(drop=True)

        self.data_dir = Path(__file__).parent.parent / data_dir
        self.sample_rate = sample_rate
        self.chunk_samples = int(chunk_duration * sample_rate)

    # -------------------------------------------------------------------------
    def __len__(self):
        return len(self.data)

    # -------------------------------------------------------------------------
    def __getitem__(self, idx):

        row = self.data.iloc[idx]
        track_id = str(row['id'])

        wav_path   = self.data_dir / "wav"    / f"{track_id}.wav"
        label_path = self.data_dir / "labels" / f"labels{track_id}.csv"

        # ── Caricamento audio ────────────────────────────────────────────────
        with sf.SoundFile(wav_path) as f:
            total_samples = len(f)
            orig_sr = f.samplerate

        if total_samples > self.chunk_samples:
            start_frame = random.randint(0, total_samples - self.chunk_samples)
        else:
            start_frame = 0

        with sf.SoundFile(wav_path) as f:
            f.seek(start_frame)
            orig_chunk_samples = int(self.chunk_samples * orig_sr / self.sample_rate)
            chunk_np = f.read(orig_chunk_samples, dtype='float32', always_2d=True)

        if chunk_np.shape[0] < orig_chunk_samples:
            pad_length = int(orig_chunk_samples - chunk_np.shape[0])
            chunk_np = np.pad(chunk_np, ((0, pad_length), (0, 0)), mode='constant')

        waveform = torch.tensor(chunk_np.T, dtype=torch.float32)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if orig_sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=orig_sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        # ── Costruzione delle matrici di label ───────────────────────────────
        ORIG_SR    = 44100
        HOP_LENGTH = 512
        MIDI_MIN   = 33     # A1
        MIDI_MAX   = 116    # C8
        NUM_NOTES  = MIDI_MAX - MIDI_MIN + 1  # 84

        df_labels = pd.read_csv(label_path)
        num_frames = self.chunk_samples // HOP_LENGTH

        # Tutte e tre le matrici: (num_frames, 84)
        piano_roll  = np.zeros((num_frames, NUM_NOTES), dtype=np.float32)
        onset_roll  = np.zeros((num_frames, NUM_NOTES), dtype=np.float32)
        offset_roll = np.zeros((num_frames, NUM_NOTES), dtype=np.float32)

        start_cqt_frame = start_frame // HOP_LENGTH

        for _, label_row in df_labels.iterrows():

            note_start = int(label_row['start_time']) // HOP_LENGTH
            note_end   = int(label_row['end_time'])   // HOP_LENGTH

            local_start = note_start - start_cqt_frame
            local_end   = note_end   - start_cqt_frame

            if local_end <= 0 or local_start >= num_frames:
                continue

            local_start_c = max(0, local_start)
            local_end_c   = min(num_frames, local_end)

            note = int(label_row['note'])
            if note < MIDI_MIN or note > MIDI_MAX:
                continue
            note_idx = note - MIDI_MIN

            # Piano roll: 1 per tutti i frame della nota
            piano_roll[local_start_c:local_end_c, note_idx] = 1.0

            # Onset: 1 solo al primo frame (se cade nel chunk)
            if local_start >= 0:
                onset_roll[local_start, note_idx] = 1.0

            # Offset: 1 solo all'ultimo frame (se cade nel chunk)
            last_frame = local_end - 1
            if 0 <= last_frame < num_frames:
                offset_roll[last_frame, note_idx] = 1.0

        return {
            "waveform": waveform,
            "labels":   torch.tensor(piano_roll,  dtype=torch.float32),
            "onsets":   torch.tensor(onset_roll,  dtype=torch.float32),
            "offsets":  torch.tensor(offset_roll, dtype=torch.float32),
            "id":       track_id
        }


# ── Test ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    dataset = MusicNetPianoDataset(split="train")
    print(f"Tracce training: {len(dataset)}")

    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Waveform : {sample['waveform'].shape}")   # (1, chunk_samples)
        print(f"Labels   : {sample['labels'].shape}")     # (T, 84)
        print(f"Onsets   : {sample['onsets'].shape}")     # (T, 84)
        print(f"Offsets  : {sample['offsets'].shape}")    # (T, 84)
        print(f"Track id : {sample['id']}")

        # Sanity check: ogni onset implica almeno un frame attivo nel piano roll
        has_onset  = sample['onsets'].sum().item()
        has_active = sample['labels'].sum().item()
        print(f"Frame attivi: {has_active:.0f} | Onset frames: {has_onset:.0f}")