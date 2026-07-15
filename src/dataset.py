
import torch
import torchaudio
import pandas as pd
import numpy as np
import random
import math
from torch.utils.data import Dataset
from pathlib import Path
import soundfile as sf
import pretty_midi

from utils import load_config


class MaestroDataset(Dataset):

    def __init__(self,
                 csv_file=None,
                 data_dir=None,
                 split='train',
                 chunk_duration=None,
                 sample_rate=None,
                 aug_config=None,
                 config_path="configs/config.yaml",
                 val_seed=42):

        cfg            = load_config(config_path)
        self.config    = cfg
        csv_file       = csv_file       or cfg['dataset']['csv_file']
        data_dir       = data_dir       or cfg['dataset']['data_dir']
        chunk_duration = chunk_duration or cfg['dataset']['chunk_duration']
        sample_rate    = sample_rate    or cfg['dataset']['sample_rate']

        project_root  = Path(__file__).parent.parent
        csv_path      = project_root / csv_file
        self.data_dir = project_root / data_dir

        df = pd.read_csv(csv_path)
        self.data          = df[df['split'] == split].reset_index(drop=True)
        self.split          = split
        self.sample_rate    = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_samples  = int(chunk_duration * sample_rate)

        self.aug_config = aug_config if (split == 'train' and aug_config is not None) else None

        print(f"Dataset '{split}' loaded: {len(self.data)} tracks.")

        HOP_LENGTH = cfg['dataset']['hop_length']
        MIDI_MIN   = cfg['dataset']['midi_min']
        MIDI_MAX   = cfg['dataset']['midi_max']
        NUM_NOTES  = MIDI_MAX - MIDI_MIN + 1
        frame_rate = sample_rate / HOP_LENGTH

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

        HOP_LENGTH = cfg['dataset']['hop_length']
        MIDI_MIN   = cfg['dataset']['midi_min']
        MIDI_MAX   = cfg['dataset']['midi_max']
        NUM_NOTES  = MIDI_MAX - MIDI_MIN + 1

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

        if label_start > 0 and label_start <= len(full_roll):
            prev_frame = full_roll[label_start - 1:label_start]
        else:
            prev_frame = np.zeros((1, NUM_NOTES), dtype=np.float32)

        if label_end < len(full_roll):
            next_frame = full_roll[label_end:label_end + 1]
        else:
            next_frame = np.zeros((1, NUM_NOTES), dtype=np.float32)

        prev_extended = np.vstack([prev_frame, piano_roll[:-1]])
        onset_roll    = np.clip(piano_roll - prev_extended, 0, 1)

        next_extended = np.vstack([piano_roll[1:], next_frame])
        offset_roll   = np.clip(piano_roll - next_extended, 0, 1)

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
            'waveform': waveform,
            'labels':   torch.tensor(piano_roll,  dtype=torch.float32),
            'onsets':   torch.tensor(onset_roll,  dtype=torch.float32),
            'offsets':  torch.tensor(offset_roll, dtype=torch.float32),
            'id':       track_id,
        }

    def _shift_piano_roll(self, piano_roll, n_steps, num_notes):
        shifted = np.zeros_like(piano_roll)
        for note_idx in range(num_notes):
            new_idx = note_idx + n_steps
            if 0 <= new_idx < num_notes:
                shifted[:, new_idx] = piano_roll[:, note_idx]
        return shifted

    def _apply_augmentation(self, waveform):
        cfg = self.aug_config
        pitch_shift_steps = 0

        if random.random() < cfg.get('pitch_shift_prob', 0.0):
            max_steps = int(cfg.get('pitch_shift_max_steps', 2))
            possible  = [s for s in range(-max_steps, max_steps + 1) if s != 0]
            pitch_shift_steps = random.choice(possible)
            waveform = torchaudio.functional.pitch_shift(
                waveform, sample_rate=self.sample_rate, n_steps=float(pitch_shift_steps))
        elif random.random() < cfg.get('detuning_prob', 0.0):
            detuning = random.uniform(-cfg.get('detuning_max_steps', 0.5),
                                       cfg.get('detuning_max_steps', 0.5))
            waveform = torchaudio.functional.pitch_shift(
                waveform, sample_rate=self.sample_rate, n_steps=detuning)

        if random.random() < cfg.get('gain_prob', 0.0):
            waveform = waveform * random.uniform(cfg.get('gain_min', 0.5), cfg.get('gain_max', 1.2))

        if random.random() < cfg.get('noise_prob', 0.0):
            nl = random.uniform(cfg.get('noise_min', 0.001), cfg.get('noise_max', 0.005))
            waveform = waveform + nl * torch.randn_like(waveform)

        if random.random() < cfg.get('reverb_prob', 0.0):
            rt60   = random.uniform(cfg.get('reverb_rt60_min', 0.3), cfg.get('reverb_rt60_max', 1.5))
            ir_len = int(rt60 * 1.2 * self.sample_rate)
            t      = torch.linspace(0, rt60 * 1.2, ir_len)
            ir     = torch.exp(-t / (rt60 / np.log(1000.0)))
            ir     = (ir / ir.sum()).unsqueeze(0)
            waveform = torchaudio.functional.convolve(waveform, ir, mode='full')
            waveform = waveform[:, :self.chunk_samples]

        return waveform, pitch_shift_steps


if __name__ == '__main__':
    config   = load_config("configs/config.yaml")
    aug_cfg  = config.get('augmentation', {})
    aug_conf = aug_cfg if aug_cfg.get('enabled', False) else None
    ds       = MaestroDataset(split='train', aug_config=aug_conf)
    print(f"Chunk di training: {len(ds)}")
    sample   = ds[0]
    print(f"Waveform : {sample['waveform'].shape}")
    print(f"Labels   : {sample['labels'].shape}   active={sample['labels'].mean():.4f}")
    print(f"Onsets   : {sample['onsets'].shape}   active={sample['onsets'].mean():.4f}")
    print(f"Offsets  : {sample['offsets'].shape}  active={sample['offsets'].mean():.4f}")