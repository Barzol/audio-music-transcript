
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
    csv_file        : path to the csv
    data_dir        : root directory of data
    split           : train, val or test
    chunk_duration  : length in seconds of each audio chunk
    sample_rate     : audio sample rate
    val_seed        : seed used to precompute deterministic chunks for
                       non-train splits, so validation/test always see
                       the same audio segment across epochs
    '''

    def __init__(self, 
                 csv_file = "data/solo_piano.csv", 
                 data_dir="data/raw", 
                 split='train', 
                 chunk_duration=5.0, 
                 sample_rate=22050,
                 val_seed=42
                 ):
        
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

        print("Pre-computing label tables...")
        self.labels_cache = {}
        for row_idx, row in self.data.iterrows():
            track_id = str(row['id'])
            label_path = self.data_dir / "labels" / f"labels{track_id}.csv"
            self.labels_cache[track_id] = pd.read_csv(label_path)
        print("Label tables ready.")

    def __len__(self):
        return len(self.index)


    def __getitem__(self, idx):
        
        '''
        here we load one training example identified by the index 'idx'

        returns :
            'waveform'  : FloatTensor of shape (1, chunk_samples)
            'labels'    : FloatTensor of shape (num_cqt_frames, num_pitches)
            'id'        : track identifier
        '''

        row_idx, chunk_idx = self.index[idx]
        row = self.data.iloc[row_idx]

        track_id = str(row['id'])

        wav_path = self.data_dir / "wav" / f"{track_id}.wav"


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
            chunk_np = f.read(
                orig_chunk_samples, 
                dtype='float32', 
                always_2d=True
                )
            
        if chunk_np.shape[0] < orig_chunk_samples:
            pad_length = int(orig_chunk_samples - chunk_np.shape[0])
            chunk_np = np.pad(chunk_np, ((0,pad_length), (0,0)), mode='constant')

        waveform = torch.tensor(chunk_np.T, dtype=torch.float32) 

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if orig_sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=orig_sr, 
                new_freq=self.sample_rate
            )
            waveform = resampler(waveform)



        ORIG_SR = 44100
        HOP_LENGTH = 512

        MIDI_MIN = 33
        MIDI_MAX = 116
        NUM_NOTES = MIDI_MAX - MIDI_MIN +1

        df = self.labels_cache[track_id]

        num_frames = self.chunk_samples // HOP_LENGTH

        piano_roll = np.zeros((num_frames, NUM_NOTES), dtype=np.float32)

        scale = self.sample_rate / ORIG_SR

        start_cqt_frame = int(start_frame * scale) // HOP_LENGTH

        for _, label_row in df.iterrows():

            note_start = int(label_row['start_time'] * scale) // HOP_LENGTH
            note_end = int(label_row['end_time'] * scale) // HOP_LENGTH

            local_start = note_start - start_cqt_frame            
            local_end   = note_end   - start_cqt_frame

            if local_end <= 0 or local_start >= num_frames :
                continue

            local_start = max(0, local_start)
            local_end = min(num_frames, local_end)

            note = int(label_row['note'])
            if note < MIDI_MIN or note > MIDI_MAX:
                continue

            note_idx = note - MIDI_MIN
            piano_roll[local_start:local_end, note_idx] = 1.0

        chunk_labels = torch.tensor(piano_roll, dtype=torch.float32)

        return {
            "waveform": waveform,
            "labels": chunk_labels, 
            "id": track_id
        }


if __name__ == "__main__":

    dataset = MusicNetPianoDataset(split="train")
    print("------- TEST ------- ")
    print(f"Chunk di training trovati: {len(dataset)}")

    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Waveform shape : {sample['waveform'].shape}")
        print(f"Labels shape   : {sample['labels'].shape}")
        print(f"Track id       : {sample['id']}")