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
import pretty_midi
from utils import load_config

class MusicNetPianoDataset(Dataset):

    '''
    csv_file        : path to the csv
    data_dir        : root directory of data
    split           : train or test
    chunk_duration  : length in seconds of each audio chunk
    sample_rate     : audio sample rate
    '''
    
    config = load_config("configs/config.yaml")

    def __init__(self, 
                 csv_file = config["dataset"]["csv_file"], 
                 data_dir = config["dataset"]["data_dir"], 
                 split='train', 
                 chunk_duration=config["dataset"]["chunk_duration"], 
                 sample_rate=config["dataset"]["sample_rate"],
                 ):
        
        # csv from absolute path in config
        df = pd.read_csv(csv_file)
        
        # filter split
        self.data = df[df['split'] == split].reset_index(drop=True)

        # data_dir absoulte path
        self.data_dir = Path(data_dir)
        
        # parameters
        self.sample_rate = sample_rate
        self.chunk_samples = int(chunk_duration * sample_rate)

        print(f"Dataset {split} loaded: {len(self.data)} tracks.")

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

        track_id = Path(row['audio_filename']).stem

        # build full paths to the audio file
        wav_path = self.data_dir / row['audio_filename']
        midi_path = self.data_dir / row['midi_filename']

        # ---------- Audio loading ----------
        info = sf.info(wav_path)
        total_samples = info.frames
        orig_sr = info.samplerate

        orig_chunk_samples = int(self.chunk_samples * orig_sr / self.sample_rate)

        # choose a random point for extracting 5 seconds
        if total_samples > orig_chunk_samples:
            start_frame = random.randint(0, total_samples - orig_chunk_samples)
        else:
            start_frame = 0

        # load only the 5-second chunk 
        # 'with' calls automatically two methods
        with sf.SoundFile(wav_path) as f:
            f.seek(start_frame)
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
                orig_sr, 
                self.sample_rate
            )
            waveform = resampler(waveform)


        # ---------- Label loading and conversion ----------
        
        start_time_sec = start_frame / orig_sr
        end_time_sec = start_time_sec + (self.chunk_samples / self.sample_rate)

        HOP_LENGTH = self.config['dataset']['hop_length']
        MIDI_MIN = self.config['dataset']['midi_min']
        MIDI_MAX = self.config['dataset']['midi_max']
        NUM_NOTES = MIDI_MAX - MIDI_MIN +1
        
        # define frame rate
        frame_rate = self.sample_rate / HOP_LENGTH

        # compute total number of CQT frames for chunk
        num_frames = self.chunk_samples // HOP_LENGTH

        # creates an empty piano roll
        piano_roll = np.zeros((num_frames, NUM_NOTES), dtype=np.float32)

        # convert audio start position to CQT frame index
        pm = pretty_midi.PrettyMIDI(str(midi_path))

        # 
        if len(pm.instruments) > 0:
            piano = pm.instruments[0]
            for note in piano.notes:
                # Verifica se la nota cade nel chunk
                if note.end > start_time_sec and note.start < end_time_sec:
                    l_start = int((note.start - start_time_sec) * frame_rate)
                    l_end = int((note.end - start_time_sec) * frame_rate)
                    
                    l_start = max(0, l_start)
                    l_end = min(num_frames, l_end)
                    
                    if MIDI_MIN <= note.pitch <= MIDI_MAX:
                        piano_roll[l_start:l_end, note.pitch - MIDI_MIN] = 1.0
                        
        
        return {
            "waveform": waveform,
            "labels": torch.tensor(piano_roll, dtype=torch.float32), 
            "id": track_id
        }


# --- TEST ----
if __name__ == "__main__":
    #
    # test dataset on a block
    # insert here

    dataset = MusicNetPianoDataset(split="train")
    print("------- TEST ------- ")
    print(f"Training tracks found: {len(dataset)}")

    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Waveform shape : {sample['waveform'].shape}")   # expect (1, 110250)
        print(f"Labels shape   : {sample['labels'].shape}")     # expect (215, num_pitches)
        print(f"Track id       : {sample['id']}")
    