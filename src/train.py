# train.py  -  Phase 3: CNN + BiLSTM + Multi-Output
#
# Rispetto alla Phase 2 / Phase 3 Exp 10-13:
#   - extract_features() al posto di extract_cqt() -> supporta CQT, STFT, Mel
#   - get_input_features() ricava input_features dal tipo di feature scelto
#   - aug_config passato al dataset (attivo solo se augmentation.enabled: true)
#   - multi_output flag: se False, allena solo pitch (onset/offset ignorati)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import MusicNetPianoDataset
from model import PianoTranscriptArchitecture
from utils import (
    extract_features, get_input_features,
    get_device, set_seed,
    save_checkpoint, load_config,
    time_start, time_stop, print_time,
)

import numpy as np
from plots import plot_loss_curve
from report import start_run, log_epoch, end_training


def compute_loss(model_out, labels, onsets, offsets,
                 criterion_pitch, criterion_onset, criterion_offset,
                 w_pitch, w_onset, w_offset, multi_output, device):
    """
    Calcola la loss totale.
    Se multi_output=False, usa solo la loss del pitch.
    """
    logit_pitch, logit_onset, logit_offset = model_out

    T = min(logit_pitch.size(1), labels.size(1))
    logit_pitch  = logit_pitch[:, :T, :]
    labels       = labels[:, :T, :]

    loss = w_pitch * criterion_pitch(logit_pitch, labels)

    if multi_output:
        logit_onset  = logit_onset[:, :T, :]
        logit_offset = logit_offset[:, :T, :]
        onsets       = onsets[:, :T, :]
        offsets      = offsets[:, :T, :]
        loss += w_onset  * criterion_onset(logit_onset,  onsets)
        loss += w_offset * criterion_offset(logit_offset, offsets)

    return loss


def train():

    start_time = time_start()

    config = load_config("configs/config.yaml")
    start_run(config)
    set_seed(42)

    device       = get_device()
    feat_config     = config['features']
    sr           = config['dataset']['sample_rate']
    multi_output = config['model'].get('multi_output', False)

    print(f"Training on : {device}")
    print(f"Feature type: {feat_config['type'].upper()}")
    print(f"Multi-output: {multi_output}")

    # Augmentation
    aug_config    = config.get('augmentation', {})
    aug_config = aug_config if aug_config.get('enabled', False) else None
    if aug_config:
        print("Augmentation: ON")

    # Dataset e DataLoader
    train_dataset = MusicNetPianoDataset(
        csv_file=config['dataset']['csv_file'],
        data_dir=config['dataset']['data_dir'],
        split='train',
        chunk_duration=config['dataset']['chunk_duration'],
        sample_rate=sr,
        aug_config=aug_config,
    )
    val_dataset = MusicNetPianoDataset(
        csv_file=config['dataset']['csv_file'],
        data_dir=config['dataset']['data_dir'],
        split='val',
        chunk_duration=config['dataset']['chunk_duration'],
        sample_rate=sr,
        aug_config=None,  # mai augmentare la validation
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(
        val_dataset, batch_size=config['training']['batch_size'], shuffle=False)

    print(f"Tracce - train: {len(train_dataset)} | val: {len(val_dataset)}")

    # Modello (input_features derivato automaticamente dal tipo di feature)
    input_features = get_input_features(feat_config, sr=sr)
    print(f"input_features: {input_features}")

    model = PianoTranscriptArchitecture(
        input_features=input_features,
        dropout=config['model']['dropout'],
        hidden_size=config['model']['hidden_size'],
        lstm_layers=config['model']['lstm_layers'],
    ).to(device)

    # Loss
    pw_pitch  = torch.ones(84).to(device) * config['training']['pos_weight_pitch']
    pw_onset  = torch.ones(84).to(device) * config['training']['pos_weight_onset']
    pw_offset = torch.ones(84).to(device) * config['training']['pos_weight_offset']

    criterion_pitch  = nn.BCEWithLogitsLoss(pos_weight=pw_pitch)
    criterion_onset  = nn.BCEWithLogitsLoss(pos_weight=pw_onset)
    criterion_offset = nn.BCEWithLogitsLoss(pos_weight=pw_offset)

    w_pitch  = config['training']['loss_weight_pitch']
    w_onset  = config['training']['loss_weight_onset']  if multi_output else 0.0
    w_offset = config['training']['loss_weight_offset'] if multi_output else 0.0
    grad_clip = config['training'].get('grad_clip', 1.0)

    # Ottimizzatore e scheduler
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min',
        patience=config['training']['scheduler_patience'],
        factor=config['training']['scheduler_factor'],
    )

    epochs        = config['training']['epochs']
    best_val_loss = float('inf')
    train_losses  = []
    val_losses    = []

    for epoch in range(epochs):
        t_epoch = time_start()

        # Train
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            waveforms = batch["waveform"]
            labels    = batch["labels"].to(device)
            onsets    = batch["onsets"].to(device)
            offsets   = batch["offsets"].to(device)

            inputs = torch.stack(
                [extract_features(w, feat_config, sr=sr) for w in waveforms]
            ).to(device)

            optimizer.zero_grad()
            out  = model(inputs)
            loss = compute_loss(out, labels, onsets, offsets,
                                criterion_pitch, criterion_onset, criterion_offset,
                                w_pitch, w_onset, w_offset, multi_output, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                waveforms = batch["waveform"]
                labels    = batch["labels"].to(device)
                onsets    = batch["onsets"].to(device)
                offsets   = batch["offsets"].to(device)

                inputs = torch.stack(
                    [extract_features(w, feat_config, sr=sr) for w in waveforms]
                ).to(device)

                out  = model(inputs)
                loss = compute_loss(out, labels, onsets, offsets,
                                    criterion_pitch, criterion_onset, criterion_offset,
                                    w_pitch, w_onset, w_offset, multi_output, device)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint(
                {
                    'state_dict': model.state_dict(),
                    'optimizer':  optimizer.state_dict(),
                    'epoch':      epoch + 1,
                    'loss':       best_val_loss,
                },
                filename=config['training']['checkpoint_path'],
            )
            print(f"  -> Nuovo best model (val_loss: {best_val_loss:.4f})")

        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time_stop(t_epoch)
        print(f"Epoch {epoch+1}/{epochs}  |  "
              f"Train: {avg_train_loss:.4f}  |  Val: {avg_val_loss:.4f}  |  "
              f"LR: {current_lr:.6f}")
        log_epoch(epoch, avg_train_loss, avg_val_loss, current_lr, epoch_time)

    print_time(time_stop(start_time))
    end_training()

    np.save('checkpoints/train_losses.npy', np.array(train_losses))
    np.save('checkpoints/val_losses.npy',   np.array(val_losses))
    plot_loss_curve(train_losses, val_losses)


if __name__ == "__main__":
    train()