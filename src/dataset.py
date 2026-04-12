# This file defines the Dataset Class
# 3 methods :
#   - __init__ : loads metadata and stores configuration
#   - __len__ : returns the number of tracks in the train/test split
#   - __getitem__ : loads a random 5 second audio chunk and its aligned labels

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

    '''
    csv_file        : path to the csv
    data_dir        : root directory of data
    split           : train or test
    chunk_duration  : length in seconds of each audio chunk
    sample_rate     : audio sample rate
    '''

    # load hyperparameters from the config file
    config = load_config("configs/config.yaml")

    def __init__(self, 
                 csv_file = config['dataset']['csv_file'], 
                 data_dir=config['dataset']['data_dir'], 
                 split='train', 
                 chunk_duration=config['dataset']['chunk_duration'], 
                 sample_rate=config['dataset']['sample_rate']
                 ):
        
        project_root = Path(__file__).parent.parent
        csv_path = project_root / csv_file
        self.data_dir = project_root / data_dir

        # Reads the .csv and filters only for 'train' and 'test'
        df = pd.read_csv(csv_path)
        self.data = df[df['split'] == split].reset_index(drop=True)

        # store configuration so __getitem__ can access them
        self.data_dir = Path(__file__).parent.parent / data_dir
        self.sample_rate = sample_rate

        # computes how many audio samples correspond to one chunk
        self.chunk_samples = int(chunk_duration * sample_rate)



# ---------------------------------------------------------------------------
    def __len__(self):
        return len(self.data)

# ---------------------------------------------------------------------------

    def __getitem__(self, idx):
        
        '''
        here we load one training example identified by the index 'idx'

        returns :
            'waveform'  : FloatTensor of shape (1, chunk_samples)
            'labels'    : FloatTensor of shape (num_cqt_frames, num_pitches)
            'id'        : track identifier
        '''

        # retrieve metadata
        row = self.data.iloc[idx]

        track_id = str(row['id'])

        # build full paths to the audio file
        wav_path = self.data_dir / "wav" / f"{track_id}.wav"
        label_path = self.data_dir / "labels" / f"labels{track_id}.csv"

        # ---------- Audio loading ----------


        # soundfile reads only the file header
        with sf.SoundFile(wav_path) as f:
            total_samples = len(f)      # total number of samples of the file
            orig_sr = f.samplerate    # original sr

        # choose a random point for extracting 5 seconds
        if total_samples > self.chunk_samples:
            start_frame = random.randint(0, total_samples - self.chunk_samples)
        else:
            start_frame = 0

        # load only the 5-second chunk 
        # 'with' calls automatically two methods
        with sf.SoundFile(wav_path) as f:
            f.seek(start_frame)
            orig_chunk_samples = int(self.chunk_samples * orig_sr / self.sample_rate)
            chunk_np = f.read(
                orig_chunk_samples, 
                dtype='float32', 
                always_2d=True
                )
            
        # padding, if the file is shorter we have to pad with zeros
        # to mantain the correct length
        if chunk_np.shape[0] < orig_chunk_samples:
            pad_length = int(orig_chunk_samples - chunk_np.shape[0])
            chunk_np = np.pad(chunk_np, ((0,pad_length), (0,0)), mode='constant')

        # convert to torch tensor and transpose to (channels, samples)
        waveform = torch.tensor(chunk_np.T, dtype=torch.float32) 

        # Mono conversion -> it averages all audio channels
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # if the sampling rate changes, this re-samples at 22.05 kHz
        if orig_sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=orig_sr, 
                new_freq=self.sample_rate
            )
            waveform = resampler(waveform)


        # ---------- Label loading and conversion ----------

        config = self.config
        HOP_LENGTH = config['dataset']['hop_length']

        MIDI_MIN = 33   # note A1
        MIDI_MAX = 116  # note C8
        NUM_NOTES = MIDI_MAX - MIDI_MIN + 1

        # read csv
        df = pd.read_csv(label_path)

        # compute total number of CQT frames for chunk
        num_frames = self.chunk_samples // HOP_LENGTH

        # creates an empty piano roll
        piano_roll = np.zeros((num_frames, NUM_NOTES), dtype=np.float32)

        # convert audio start position to CQT frame index
        start_cqt_frame = start_frame // HOP_LENGTH

        for _, label_row in df.iterrows():

            note_start = int(label_row['start_time']) // HOP_LENGTH
            note_end = int(label_row['end_time']) // HOP_LENGTH

            local_start = note_start - start_cqt_frame            
            local_end   = note_end   - start_cqt_frame

            if local_end <= 0 or local_start >= num_frames :
                continue

            # clamp to valid range
            local_start = max(0, local_start)
            local_end = min(num_frames, local_end)

            note = int(label_row['note'])
            # skip notes outside A1-C8 range
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


# --- TEST ----
if __name__ == "__main__":
    #
    # test dataset on a block
    # insert here

    dataset = MusicNetPianoDataset(split="train")
    print("------- TEST ------- ")
    print(f"Tracce di training trovate: {len(dataset)}")

    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Waveform shape : {sample['waveform'].shape}")   # expect (1, 110250)
        print(f"Labels shape   : {sample['labels'].shape}")     # expect (215, num_pitches)
        print(f"Track id       : {sample['id']}")
    