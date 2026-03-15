import torch
import random
import numpy as np
import os
import yaml
import librosa

# verifies cuda
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# fixed seed for evaluate the experiments
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available() : torch.cuda.manual_seed_all(seed)

# Saves the weights and the optimizer status
def save_checkpoint(state, filename="my_checkpoint.pth", dir_path="checkpoints"):
    os.makedirs(dir_path, exist_ok=True)
    filepath = os.path.join(dir_path, filename)
    torch.save(state, filepath)
    print(f"Checkpoint saved at {filepath}")

# Loads the weights of the model
def load_checkpoint(checkpoint_path, model, optimizer=None, device="cpu"):
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['state_dict'])
    
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
        
    print("Checkpoint loaded successfully.")
    return checkpoint 

# Loads hyperparameters from YAML
def load_config(config_path="configs/config.yaml"):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
    
# ---- EXTRACTION OF CQT FEATURES ---- 
# this extracts the CQT from audio. 
# - Converts the tensor in an array for librosa
# - Calculates CQT
# - takes the magnitude and ignore phase
# - Converts amplitude in dB
# - Returns the tensor

def extract_cqt(
        waveform,
        sr = 22050,
        hop_length = 512,
        n_bins = 84,
        bins_per_octave = 12
):
    if isinstance(waveform,torch.Tensor):
        waveform = waveform.squeeze().numpy()

    cqt_complex = librosa.cqt(
        y = waveform,
        sr = sr,
        hop_length = hop_length,
        fmin = librosa.note_to_hz('A0'),
        n_bins = n_bins,
        bins_per_octave = bins_per_octave
    )

    cqt_mag = np.abs(cqt_complex)

    cqt_db = librosa.amplitude_to_db(cqt_mag, ref=np.max)

    # the tensor is [frame, 84 notes]
    return torch.tensor(cqt_db, dtype=torch.float32).T
    

