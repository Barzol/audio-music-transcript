
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
    split           : train or test
    chunk_duration  : length in seconds of each audio chunk
    sample_rate     : audio sample rate
    '''

    def __init__(self, 
                 csv_file = "data/solo_piano.csv", 
                 data_dir="data/raw", 
                 split='train', 
                 chunk_duration=5.0, 
                 sample_rate=22050
                 ):
        
        project_root = Path(__file__).parent.parent
        csv_path = project_root / csv_file
        self.data_dir = project_root / data_dir

        df = pd.read_csv(csv_path)
        self.data = df[df['split'] == split].reset_index(drop=True)

        self.data_dir = Path(__file__).parent.parent / data_dir
        self.sample_rate = sample_rate

        self.chunk_samples = int(chunk_duration * sample_rate)

    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        
        '''
        here we load one training example identified by the index 'idx'

        returns :
            'waveform'  : FloatTensor of shape (1, chunk_samples)
            'labels'    : FloatTensor of shape (num_cqt_frames, num_pitches)
            'id'        : track identifier
        '''

        row = self.data.iloc[idx]

        track_id = str(row['id'])

        wav_path = self.data_dir / "wav" / f"{track_id}.wav"
        label_path = self.data_dir / "labels" / f"labels{track_id}.csv"



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

        df = pd.read_csv(label_path)

        num_frames = self.chunk_samples // HOP_LENGTH

        piano_roll = np.zeros((num_frames, NUM_NOTES), dtype=np.float32)

        start_cqt_frame = start_frame // HOP_LENGTH

        for _, label_row in df.iterrows():

            note_start = int(label_row['start_time']) // HOP_LENGTH
            note_end = int(label_row['end_time']) // HOP_LENGTH

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
    print(f"Tracce di training trovate: {len(dataset)}")

    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Waveform shape : {sample['waveform'].shape}")
        print(f"Labels shape   : {sample['labels'].shape}")
        print(f"Track id       : {sample['id']}")
    