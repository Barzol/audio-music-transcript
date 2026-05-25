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
import math

class MaestroDataset(Dataset):

    '''
    csv_file        : path to the csv
    data_dir        : root directory of data
    split           : train or test
    chunk_duration  : length in seconds of each audio chunk
    sample_rate     : audio sample rate
    '''
    
    config = load_config()

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
        
        # costanti — aggiungile qui
        HOP_LENGTH = self.config['dataset']['hop_length']
        MIDI_MIN   = self.config['dataset']['midi_min']
        MIDI_MAX   = self.config['dataset']['midi_max']
        NUM_NOTES  = MIDI_MAX - MIDI_MIN + 1
        frame_rate = self.sample_rate / HOP_LENGTH

        print("Pre-computing piano rolls...")
        self.piano_rolls = {}
        
        for _, row in self.data.iterrows():
            midi_path = self.data_dir / row['midi_filename']
            pm = pretty_midi.PrettyMIDI(str(midi_path))
            
            total_frames = int(pm.get_end_time() * frame_rate) + 1
            roll = np.zeros((total_frames, NUM_NOTES), dtype=np.float32)
            
            if len(pm.instruments) > 0:
                for note in pm.instruments[0].notes:
                    f_start = int(note.start * frame_rate)
                    f_end   = min(int(note.end * frame_rate), total_frames)
                    if MIDI_MIN <= note.pitch <= MIDI_MAX:
                        roll[f_start:f_end, note.pitch - MIDI_MIN] = 1.0
 
            self.piano_rolls[row['midi_filename']] = roll
 
        print("Piano rolls ready.")

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
            pad_length = orig_chunk_samples - chunk_np.shape[0]
            chunk_np   = np.pad(chunk_np, ((0, pad_length), (0, 0)), mode='constant')
            
        # convert to torch tensor and transpose to (channels, samples)
        waveform = torch.tensor(chunk_np.T, dtype=torch.float32) 

        # mono conversion
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
 
        # resample if needed
        if orig_sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_sr, self.sample_rate)
            waveform  = resampler(waveform)


        # ---------- Labels ----------
        HOP_LENGTH = self.config['dataset']['hop_length']
        MIDI_MIN   = self.config['dataset']['midi_min']
        MIDI_MAX   = self.config['dataset']['midi_max']
        NUM_NOTES  = MIDI_MAX - MIDI_MIN + 1
 
        frame_rate     = self.sample_rate / HOP_LENGTH
        num_frames     = 1 + math.floor(self.chunk_samples / HOP_LENGTH)
        start_time_sec = start_frame / orig_sr          # correct time alignment
        label_start    = int(start_time_sec * frame_rate)
        label_end      = label_start + num_frames
 
        full_roll  = self.piano_rolls[row['midi_filename']]
        piano_roll = full_roll[label_start:min(label_end, len(full_roll))]
 
        if len(piano_roll) < num_frames:
            pad        = np.zeros((num_frames - len(piano_roll), NUM_NOTES), dtype=np.float32)
            piano_roll = np.vstack([piano_roll, pad])
 
        return {
            "waveform": waveform,
            "labels":   torch.tensor(piano_roll, dtype=torch.float32),
            "id":       track_id
        }


# --- TEST ----
if __name__ == "__main__":
    #
    # test dataset on a block
    # insert here

    dataset = MaestroDataset(split="train")
    sample  = dataset[0]
 
    labels   = sample['labels']
    waveform = sample['waveform']
 
    print(f"Waveform shape : {waveform.shape}")
    print(f"Labels shape   : {labels.shape}")
    print(f"Active frames  : {(labels.sum(dim=1) > 0).sum().item()} / {labels.shape[0]}")
    print(f"Active ratio   : {labels.mean().item():.4f}")
    print(f"Max simultaneous notes : {labels.sum(dim=1).max().item()}")