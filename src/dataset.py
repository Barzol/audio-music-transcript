# dataset.py  –  Phase 3: CNN + BiLSTM + Multi-Output
#
# Rispetto alla Phase 2:
#   - Integrata pipeline di data augmentation (pitch shift, detuning, gain,
#     rumore gaussiano, reverb sintetico) — adattata dal codice esterno
#   - _shift_piano_roll applicato anche a onset_roll e offset_roll
#   - aug_config=None disattiva l'augmentation; ignorata su split != 'train'
#   - Fix allineamento label: start_cqt_frame calcolato con fattore di scala
#     sample_rate / ORIG_SR (necessario per resampling 44100 -> 22050)

import torch
import torchaudio
import pandas as pd
import numpy as np
import random
from torch.utils.data import Dataset
from pathlib import Path
import soundfile as sf

from utils import load_config


class MusicNetPianoDataset(Dataset):

    config = load_config("configs/config.yaml")

    def __init__(
        self,
        csv_file       = "data/solo_piano.csv",
        data_dir       = "data/raw",
        split          = 'train',
        chunk_duration = config["dataset"]["chunk_duration"],
        sample_rate    = config["dataset"]["sample_rate"],
        aug_config     = None,
    ):
        project_root  = Path(__file__).parent.parent
        csv_path      = project_root / csv_file
        self.data_dir = project_root / data_dir

        df = pd.read_csv(csv_path)
        self.data = df[df['split'] == split].reset_index(drop=True)

        self.sample_rate   = sample_rate
        self.chunk_samples = int(chunk_duration * sample_rate)
        self.aug_config    = aug_config if (split == 'train' and aug_config is not None) else None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row      = self.data.iloc[idx]
        track_id = str(row['id'])

        wav_path   = self.data_dir / "wav"    / f"{track_id}.wav"
        label_path = self.data_dir / "labels" / f"labels{track_id}.csv"

        # Audio
        with sf.SoundFile(wav_path) as f:
            total_samples = len(f)
            orig_sr       = f.samplerate

        orig_chunk_samples = int(self.chunk_samples * orig_sr / self.sample_rate)
        start_frame = 0
        if total_samples > orig_chunk_samples:
            start_frame = random.randint(0, total_samples - orig_chunk_samples)

        with sf.SoundFile(wav_path) as f:
            f.seek(start_frame)
            chunk_np = f.read(orig_chunk_samples, dtype='float32', always_2d=True)

        if chunk_np.shape[0] < orig_chunk_samples:
            pad_len  = orig_chunk_samples - chunk_np.shape[0]
            chunk_np = np.pad(chunk_np, ((0, pad_len), (0, 0)), mode='constant')

        waveform = torch.tensor(chunk_np.T, dtype=torch.float32)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if orig_sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(
                orig_freq=orig_sr, new_freq=self.sample_rate)(waveform)

        # Label
        ORIG_SR    = 44100
        HOP_LENGTH = 512
        MIDI_MIN   = 33
        MIDI_MAX   = 116
        NUM_NOTES  = MIDI_MAX - MIDI_MIN + 1

        df_labels  = pd.read_csv(label_path)
        num_frames = self.chunk_samples // HOP_LENGTH

        piano_roll  = np.zeros((num_frames, NUM_NOTES), dtype=np.float32)
        onset_roll  = np.zeros((num_frames, NUM_NOTES), dtype=np.float32)
        offset_roll = np.zeros((num_frames, NUM_NOTES), dtype=np.float32)

        scale           = self.sample_rate / ORIG_SR
        start_cqt_frame = int(start_frame * scale) // HOP_LENGTH

        for _, label_row in df_labels.iterrows():
            note_start = int(label_row['start_time'] * scale) // HOP_LENGTH
            note_end   = int(label_row['end_time']   * scale) // HOP_LENGTH
            local_start = note_start - start_cqt_frame
            local_end   = note_end   - start_cqt_frame

            if local_end <= 0 or local_start >= num_frames:
                continue
            note = int(label_row['note'])
            if note < MIDI_MIN or note > MIDI_MAX:
                continue
            note_idx = note - MIDI_MIN

            piano_roll[max(0,local_start):min(num_frames,local_end), note_idx] = 1.0
            if local_start >= 0:
                onset_roll[local_start, note_idx] = 1.0
            last_frame = local_end - 1
            if 0 <= last_frame < num_frames:
                offset_roll[last_frame, note_idx] = 1.0

        # Augmentation
        pitch_shift_steps = 0
        if self.aug_config is not None:
            waveform, pitch_shift_steps = self._apply_augmentation(waveform)

        if pitch_shift_steps != 0:
            piano_roll  = self._shift_piano_roll(piano_roll,  pitch_shift_steps, NUM_NOTES)
            onset_roll  = self._shift_piano_roll(onset_roll,  pitch_shift_steps, NUM_NOTES)
            offset_roll = self._shift_piano_roll(offset_roll, pitch_shift_steps, NUM_NOTES)

        max_val = torch.abs(waveform).max()
        if max_val > 0:
            waveform = waveform / max_val

        return {
            "waveform": waveform,
            "labels":   torch.tensor(piano_roll,  dtype=torch.float32),
            "onsets":   torch.tensor(onset_roll,  dtype=torch.float32),
            "offsets":  torch.tensor(offset_roll, dtype=torch.float32),
            "id":       track_id,
        }

    def _shift_piano_roll(self, piano_roll, n_steps, num_notes):
        """Trasla le note di n_steps semitoni. Note fuori range [0,num_notes) scartate."""
        shifted = np.zeros_like(piano_roll)
        for note_idx in range(num_notes):
            new_idx = note_idx + n_steps
            if 0 <= new_idx < num_notes:
                shifted[:, new_idx] = piano_roll[:, note_idx]
        return shifted

    def _apply_augmentation(self, waveform):
        """
        Pitch shift (int) | detuning (float, mutualmente esclusivi),
        gain, rumore gaussiano, reverb sintetico.
        Restituisce (waveform, pitch_shift_steps).
        """
        config = self.aug_config
        pitch_shift_steps = 0

        # 1. Pitch shift / detuning
        if random.random() < config.get('pitch_shift_prob', 0.0):
            max_steps = int(config.get('pitch_shift_max_steps', 2))
            possible  = [s for s in range(-max_steps, max_steps + 1) if s != 0]
            pitch_shift_steps = random.choice(possible)
            waveform = torchaudio.functional.pitch_shift(
                waveform, sample_rate=self.sample_rate, n_steps=float(pitch_shift_steps))
        elif random.random() < config.get('detuning_prob', 0.0):
            detuning = random.uniform(-config.get('detuning_max_steps', 0.5),
                                       config.get('detuning_max_steps', 0.5))
            waveform = torchaudio.functional.pitch_shift(
                waveform, sample_rate=self.sample_rate, n_steps=detuning)

        # 2. Gain
        if random.random() < config.get('gain_prob', 0.0):
            waveform = waveform * random.uniform(config.get('gain_min', 0.5), config.get('gain_max', 1.2))

        # 3. Rumore gaussiano (SNR > 40 dB a noise_max=0.005)
        if random.random() < config.get('noise_prob', 0.0):
            nl = random.uniform(config.get('noise_min', 0.001), config.get('noise_max', 0.005))
            waveform = waveform + nl * torch.randn_like(waveform)

        # 4. Reverb sintetico (IR esponenziale, RT60 in [0.3, 1.5] s)
        if random.random() < config.get('reverb_prob', 0.0):
            rt60   = random.uniform(config.get('reverb_rt60_min', 0.3), config.get('reverb_rt60_max', 1.5))
            ir_len = int(rt60 * 1.2 * self.sample_rate)
            t      = torch.linspace(0, rt60 * 1.2, ir_len)
            ir     = torch.exp(-t / (rt60 / np.log(1000.0)))
            ir     = (ir / ir.sum()).unsqueeze(0)
            waveform = torchaudio.functional.convolve(waveform, ir, mode='full')
            waveform = waveform[:, :self.chunk_samples]

        return waveform, pitch_shift_steps


if __name__ == "__main__":
    config    = load_config("configs/config.yaml")
    augconfig   = config.get('augmentation', {})
    aug_conf  = augconfig if augconfig.get('enabled', False) else None
    dataset   = MusicNetPianoDataset(split="train", aug_config=aug_conf)
    print(f"Tracce: {len(dataset)}")
    s = dataset[0]
    print(f"waveform {s['waveform'].shape}  labels {s['labels'].shape}  "
          f"onsets {s['onsets'].shape}  offsets {s['offsets'].shape}")