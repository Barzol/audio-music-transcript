

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
    split           : 'train', 'val' o 'test'
    chunk_duration  : durata in secondi di ogni chunk audio
    sample_rate     : frequenza di campionamento
    val_seed        : seed per i chunk fissi di val/test
    '''

    def __init__(self,
                 csv_file="data/solo_piano.csv",
                 data_dir="data/raw",
                 split='train',
                 chunk_duration=5.0,
                 sample_rate=22050,
                 val_seed=42):

        project_root = Path(__file__).parent.parent
        csv_path = project_root / csv_file
        self.data_dir = project_root / data_dir

        df = pd.read_csv(csv_path)
        self.data = df[df['split'] == split].reset_index(drop=True)

        self.data_dir = Path(__file__).parent.parent / data_dir
        self.split = split
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_samples = int(chunk_duration * sample_rate)

        self.index = []
        self._orig_sr_cache = {}

        for row_idx, row in self.data.iterrows():
            track_id = str(row['id'])
            wav_path = self.data_dir / "wav" / f"{track_id}.wav"

            with sf.SoundFile(wav_path) as f:
                total_samples = len(f)
                orig_sr = f.samplerate

            self._orig_sr_cache[row_idx] = orig_sr
            orig_chunk_samples = int(self.chunk_samples * orig_sr / self.sample_rate)
            n_chunks = max(1, total_samples // orig_chunk_samples)

            for chunk_idx in range(n_chunks):
                self.index.append((row_idx, chunk_idx))

        self.fixed_start_frames = {}
        if split != 'train':
            val_rng = random.Random(val_seed)
            for row_idx, chunk_idx in self.index:
                orig_sr = self._orig_sr_cache[row_idx]
                orig_chunk_samples = int(self.chunk_samples * orig_sr / self.sample_rate)
                base = chunk_idx * orig_chunk_samples
                jitter = val_rng.randint(0, max(0, orig_chunk_samples // 4))
                self.fixed_start_frames[(row_idx, chunk_idx)] = base + jitter

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):

        row_idx, chunk_idx = self.index[idx]
        row = self.data.iloc[row_idx]
        track_id = str(row['id'])

        wav_path   = self.data_dir / "wav"    / f"{track_id}.wav"
        label_path = self.data_dir / "labels" / f"labels{track_id}.csv"

        with sf.SoundFile(wav_path) as f:
            total_samples = len(f)
            orig_sr = f.samplerate

        orig_chunk_samples = int(self.chunk_samples * orig_sr / self.sample_rate)

        if self.split == 'train':
            if total_samples > orig_chunk_samples:
                start_frame = random.randint(0, total_samples - orig_chunk_samples)
            else:
                start_frame = 0
        else:
            start_frame = min(
                self.fixed_start_frames[(row_idx, chunk_idx)],
                max(0, total_samples - orig_chunk_samples)
            )

        with sf.SoundFile(wav_path) as f:
            f.seek(start_frame)
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

        ORIG_SR    = 44100
        HOP_LENGTH = 512
        MIDI_MIN   = 33
        MIDI_MAX   = 116
        NUM_NOTES  = MIDI_MAX - MIDI_MIN + 1

        df_labels = pd.read_csv(label_path)
        num_frames = self.chunk_samples // HOP_LENGTH

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

            piano_roll[local_start_c:local_end_c, note_idx] = 1.0

            if local_start >= 0:
                onset_roll[local_start, note_idx] = 1.0

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


if __name__ == "__main__":
    dataset = MusicNetPianoDataset(split="train")
    print(f"Chunk training: {len(dataset)}")

    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Waveform : {sample['waveform'].shape}")
        print(f"Labels   : {sample['labels'].shape}")
        print(f"Onsets   : {sample['onsets'].shape}")
        print(f"Offsets  : {sample['offsets'].shape}")
        print(f"Track id : {sample['id']}")

        has_onset  = sample['onsets'].sum().item()
        has_active = sample['labels'].sum().item()
        print(f"Frame attivi: {has_active:.0f} | Onset frames: {has_onset:.0f}")