# utils.py  –  Phase 3: CNN + BiLSTM + Multi-Output
#
# Aggiunto rispetto alla Phase 2:
#   - extract_stft  : STFT magnitude in dB, ritagliata al range pianistico
#   - extract_mel   : Mel spectrogram in dB
#   - extract_features : dispatcher basato su config['features']['type']
#   - get_input_features : restituisce il numero di bin per tipo di feature

import torch
import random
import numpy as np
import os
import yaml
import librosa
import time


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def load_config(config_path="configs/config.yaml"):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


# ── Estrazione feature ────────────────────────────────────────────────────────

def extract_cqt(waveform, sr=22050, hop_length=512, n_bins=84, bins_per_octave=12):
    """
    CQT logaritmico, 84 bin (A1-C8).
    Input : tensore (1, samples) o array 1D
    Output: tensore (T, n_bins) in dB
    """
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.squeeze(0).numpy()

    cqt_complex = librosa.cqt(
        y=waveform,
        sr=sr,
        hop_length=hop_length,
        fmin=librosa.note_to_hz('A1'),
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
    )
    cqt_db = librosa.amplitude_to_db(np.abs(cqt_complex), ref=np.max)
    return torch.tensor(cqt_db, dtype=torch.float32).T  # (T, n_bins)


def extract_stft(waveform, sr=22050, hop_length=512, n_fft=2048,
                 fmin=55.0, fmax=4200.0):
    """
    STFT magnitude in dB, ritagliata al range pianistico [fmin, fmax].
    fmin=55 Hz  ~ A1 (MIDI 33)
    fmax=4200 Hz ~ C8 (MIDI 116, fondamentale ~4186 Hz)

    Con n_fft=2048, sr=22050 -> risoluzione ~10.8 Hz/bin -> ~386 bin nel range
    Con n_fft=1024, sr=22050 -> risoluzione ~21.5 Hz/bin -> ~193 bin nel range

    Input : tensore (1, samples) o array 1D
    Output: tensore (T, n_freq_bins_in_range) in dB
    """
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.squeeze(0).numpy()

    stft_complex = librosa.stft(waveform, n_fft=n_fft, hop_length=hop_length)
    mag_db = librosa.amplitude_to_db(np.abs(stft_complex), ref=np.max)

    # Ritaglia al range pianistico
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    mask  = (freqs >= fmin) & (freqs <= fmax)
    mag_db = mag_db[mask, :]

    return torch.tensor(mag_db, dtype=torch.float32).T  # (T, n_freq_bins)


def extract_mel(waveform, sr=22050, hop_length=512, n_fft=2048, n_mels=128):
    """
    Mel spectrogram in dB.
    Input : tensore (1, samples) o array 1D
    Output: tensore (T, n_mels) in dB
    """
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.squeeze(0).numpy()

    mel = librosa.feature.melspectrogram(
        y=waveform, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return torch.tensor(mel_db, dtype=torch.float32).T  # (T, n_mels)


def extract_features(waveform, feat_cfg, sr=22050):
    """
    Dispatcher: chiama l'estrattore corretto in base a feat_cfg['type'].

    feat_cfg : dizionario dalla sezione config['features']
    sr       : sample rate (da config['dataset']['sample_rate'])
    """
    t          = feat_cfg['type']
    hop_length = feat_cfg.get('hop_length', 512)

    if t == 'cqt':
        return extract_cqt(
            waveform, sr=sr, hop_length=hop_length,
            n_bins=feat_cfg.get('cqt_bins', 84),
        )
    elif t == 'stft':
        return extract_stft(
            waveform, sr=sr, hop_length=hop_length,
            n_fft=feat_cfg.get('stft_n_fft', 2048),
            fmin=feat_cfg.get('stft_fmin', 55.0),
            fmax=feat_cfg.get('stft_fmax', 4200.0),
        )
    elif t == 'mel':
        return extract_mel(
            waveform, sr=sr, hop_length=hop_length,
            n_fft=feat_cfg.get('mel_n_fft', 2048),
            n_mels=feat_cfg.get('mel_n_mels', 128),
        )
    else:
        raise ValueError(f"Feature type sconosciuto: '{t}'. Usa 'cqt', 'stft' o 'mel'.")


def get_input_features(feat_cfg, sr=22050):
    """
    Restituisce il numero di bin di frequenza per il tipo di feature scelto.
    Usato per impostare input_features del modello in modo automatico.
    """
    t = feat_cfg['type']

    if t == 'cqt':
        return feat_cfg.get('cqt_bins', 84)

    elif t == 'stft':
        n_fft = feat_cfg.get('stft_n_fft', 2048)
        fmin  = feat_cfg.get('stft_fmin', 55.0)
        fmax  = feat_cfg.get('stft_fmax', 4200.0)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        return int(np.sum((freqs >= fmin) & (freqs <= fmax)))

    elif t == 'mel':
        return feat_cfg.get('mel_n_mels', 128)

    else:
        raise ValueError(f"Feature type sconosciuto: '{t}'.")


# ── Timing ────────────────────────────────────────────────────────────────────

def time_start():
    return time.time()

def time_stop(start_time):
    return time.time() - start_time

def print_time(elapsed):
    h = int(elapsed // 3600)
    m = int((elapsed % 3600) // 60)
    s = int(elapsed % 60)
    print(f"\nTotal time: {h:02d}h {m:02d}m {s:02d}s")