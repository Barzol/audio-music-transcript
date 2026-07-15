 
import torch
import random
import numpy as np
import os
import yaml
import librosa
import time
from pathlib import Path
 
 
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available() : torch.cuda.manual_seed_all(seed)
 
def save_checkpoint(state, filename="my_checkpoint.pth", dir_path="checkpoints"):
    os.makedirs(dir_path, exist_ok=True)
    filepath = os.path.join(dir_path, filename)
    torch.save(state, filepath)
    print(f"Checkpoint saved at {filepath}")
 
def load_checkpoint(checkpoint_path, model, optimizer=None, device="cpu"):
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
 
    model.load_state_dict(checkpoint['state_dict'])
 
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
 
    print("Checkpoint loaded successfully.")
    return checkpoint
 
def load_config(config_path=None):
    if config_path is None :
        config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
 
 
def extract_cqt(
        waveform,
        sr = 22050,
        hop_length = 512,
        n_bins = 84,
        bins_per_octave = 12
):
    if isinstance(waveform,torch.Tensor):
        waveform = waveform.squeeze(0).numpy()
 
    cqt_complex = librosa.cqt(
        y = waveform,
        sr = sr,
        hop_length = hop_length,
        fmin = librosa.note_to_hz('A1'),
        n_bins = n_bins,
        bins_per_octave = bins_per_octave
    )
 
    cqt_mag = np.abs(cqt_complex)
 
    cqt_db = librosa.amplitude_to_db(cqt_mag, ref=1.0)
    cqt_db = np.clip(cqt_db, a_min = -80.0, a_max = 0.0)
    cqt_db = (cqt_db + 80.0) / 80.0
 
    return torch.tensor(cqt_db, dtype=torch.float32).T
 
 
def time_start():
    start_time = time.time()
    return start_time
 
def time_stop(start_time):
    elapsed = time.time() - start_time
    return elapsed
 
def print_time(elapsed):
    hours   = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)
    print(f"\nTotal time: {hours:02d}h {minutes:02d}m {seconds:02d}s")
 
 
def generate_test_samples(track_info, plot_func, threshold):
 
    print(f"Generating {len(track_info)} Piano Rolls...")
    for info in track_info:
        plot_func(
            info['labels'],
            info['probs'],
            track_id=info['id'],
            threshold=threshold
        )
 