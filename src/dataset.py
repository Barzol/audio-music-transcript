# Dataset Class for MAESTRO - Phase 2
# Onset and offset are derived on-the-fly from the pitch roll in __getitem__
# to avoid allocating 3x float32 arrays per track in __init__.

import torch
import torchaudio
import pandas as pd
import numpy as np
import random
from torch.utils.data import Dataset
from pathlib import Path
import soundfile as sf
import pretty_midi
from utils import load_config
import math


class MaestroDataset(Dataset):

    config = load_config()

    def __init__(self,
                 csv_file=None,
                 data_dir=None,
                 split='train',
                 chunk_duration=None,
                 sample_rate=None,
                 val_seed=42):

        cfg = self.config
        csv_file       = csv_file       or cfg['dataset']['csv_file']
        data_dir       = data_dir       or cfg['dataset']['data_dir']
        chunk_duration = chunk_duration or cfg['dataset']['chunk_duration']
        sample_rate    = sample_rate    or cfg['dataset']['sample_rate']

        df = pd.read_csv(csv_file)
        self.data      = df[df['split'] == split].reset_index(drop=True)
        self.data_dir  = Path(data_dir)
        self.split          = split
        self.sample_rate    = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_samples  = int(chunk_duration * sample_rate)

        print(f"Dataset '{split}' loaded: {len(self.data)} tracks.")

        HOP_LENGTH = cfg['dataset']['hop_length']
        MIDI_MIN   = cfg['dataset']['midi_min']
        MIDI_MAX   = cfg['dataset']['midi_max']
        NUM_NOTES  = MIDI_MAX - MIDI_MIN + 1
        frame_rate = sample_rate / HOP_LENGTH

        # Only pre-compute pitch rolls (same memory footprint as Phase 1).
        # Onset and offset are derived on-the-fly in __getitem__.
        print("Pre-computing pitch piano rolls...")
        self.piano_rolls = {}

        for _, row in self.data.iterrows():
            midi_path    = self.data_dir / row['midi_filename']
            pm           = pretty_midi.PrettyMIDI(str(midi_path))
            total_frames = int(pm.get_end_time() * frame_rate) + 1

            roll = np.zeros((total_frames, NUM_NOTES), dtype=np.float32)

            if len(pm.instruments) > 0:
                for note in pm.instruments[0].notes:
                    if not (MIDI_MIN <= note.pitch <= MIDI_MAX):
                        continue
                    pitch_idx = note.pitch - MIDI_MIN
                    f_start   = int(note.start * frame_rate)
                    f_end     = min(int(note.end * frame_rate), total_frames - 1)
                    roll[f_start:f_end + 1, pitch_idx] = 1.0

            self.piano_rolls[row['midi_filename']] = roll

        print("Piano rolls ready.")

        # --- FIX 1: indice (track_idx, chunk_idx) ---
        self.index = []
        self._orig_sr_cache = {}

        for row_idx, row in self.data.iterrows():
            wav_path = self.data_dir / row['audio_filename']
            info     = sf.info(wav_path)

            self._orig_sr_cache[row_idx] = info.samplerate
            orig_chunk_samples = int(self.chunk_samples * info.samplerate / self.sample_rate)
            n_chunks = max(1, info.frames // orig_chunk_samples)

            for chunk_idx in range(n_chunks):
                self.index.append((row_idx, chunk_idx))

        # --- FIX 2: start_frame fisso per split non-train ---
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
        cfg      = self.config
        row_idx, chunk_idx = self.index[idx]
        row      = self.data.iloc[row_idx]
        track_id = Path(row['audio_filename']).stem
        wav_path = self.data_dir / row['audio_filename']

        # Audio
        info               = sf.info(wav_path)
        total_samples      = info.frames
        orig_sr            = info.samplerate
        orig_chunk_samples = int(self.chunk_samples * orig_sr / self.sample_rate)

        if self.split == 'train':
            start_frame = random.randint(0, max(0, total_samples - orig_chunk_samples))
        else:
            start_frame = min(
                self.fixed_start_frames[(row_idx, chunk_idx)],
                max(0, total_samples - orig_chunk_samples)
            )

        with sf.SoundFile(wav_path) as f:
            f.seek(start_frame)
            chunk_np = f.read(orig_chunk_samples, dtype='float32', always_2d=True)

        if chunk_np.shape[0] < orig_chunk_samples:
            pad      = orig_chunk_samples - chunk_np.shape[0]
            chunk_np = np.pad(chunk_np, ((0, pad), (0, 0)), mode='constant')

        waveform = torch.tensor(chunk_np.T, dtype=torch.float32)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if orig_sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(orig_sr, self.sample_rate)(waveform)

        # Pitch labels
        HOP_LENGTH = cfg['dataset']['hop_length']
        MIDI_MIN   = cfg['dataset']['midi_min']
        MIDI_MAX   = cfg['dataset']['midi_max']
        NUM_NOTES  = MIDI_MAX - MIDI_MIN + 1

        frame_rate     = self.sample_rate / HOP_LENGTH
        num_frames     = 1 + math.floor(self.chunk_samples / HOP_LENGTH)
        start_time_sec = start_frame / orig_sr
        label_start    = int(start_time_sec * frame_rate)
        label_end      = label_start + num_frames

        full_roll  = self.piano_rolls[row['midi_filename']]
        piano_roll = full_roll[label_start:min(label_end, len(full_roll))]

        if len(piano_roll) < num_frames:
            pad        = np.zeros((num_frames - len(piano_roll), NUM_NOTES), dtype=np.float32)
            piano_roll = np.vstack([piano_roll, pad])

        # Onset / Offset derived on-the-fly from pitch roll
        # onset[t]  = 1  when pitch[t]=1 and pitch[t-1]=0  (note starts)
        # offset[t] = 1  when pitch[t]=1 and pitch[t+1]=0  (note ends)

        # Frame immediately before chunk start (for onset detection at t=0)
        if label_start > 0 and label_start <= len(full_roll):
            prev_frame = full_roll[label_start - 1:label_start]
        else:
            prev_frame = np.zeros((1, NUM_NOTES), dtype=np.float32)

        # Frame immediately after chunk end (for offset detection at t=T-1)
        if label_end < len(full_roll):
            next_frame = full_roll[label_end:label_end + 1]
        else:
            next_frame = np.zeros((1, NUM_NOTES), dtype=np.float32)

        # onset: current frame active, previous frame inactive
        prev_extended = np.vstack([prev_frame, piano_roll[:-1]])   # (T, 84)
        onset_roll    = np.clip(piano_roll - prev_extended, 0, 1)

        # offset: current frame active, next frame inactive
        next_extended = np.vstack([piano_roll[1:], next_frame])    # (T, 84)
        offset_roll   = np.clip(piano_roll - next_extended, 0, 1)

        return {
            'waveform':      waveform,
            'labels':        torch.tensor(piano_roll,  dtype=torch.float32),
            'onset_labels':  torch.tensor(onset_roll,  dtype=torch.float32),
            'offset_labels': torch.tensor(offset_roll, dtype=torch.float32),
            'id':            track_id
        }


if __name__ == '__main__':
    ds     = MaestroDataset(split='train')
    print(f"Chunk di training: {len(ds)}")
    sample = ds[0]
    print(f"Waveform : {sample['waveform'].shape}")
    print(f"Pitch    : {sample['labels'].shape}  active={sample['labels'].mean():.4f}")
    print(f"Onset    : {sample['onset_labels'].shape}  active={sample['onset_labels'].mean():.4f}")
    print(f"Offset   : {sample['offset_labels'].shape}  active={sample['offset_labels'].mean():.4f}")