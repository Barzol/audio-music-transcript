# train.py  –  Fase 1: loop di training per il Baseline CNN
#
# Uso:
#   python train.py
#
# Legge gli iperparametri da configs/config.yaml,
# carica il dataset MusicNet (solo pianoforte),
# estrae le feature CQT e allena la BaselineCNN.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import MusicNetPianoDataset
from model import PianoTranscriptArchitecture
from utils import (
    extract_cqt, get_device, set_seed,
    save_checkpoint, load_config,
    time_start, time_stop, print_time,
)

import numpy as np
from plots import plot_loss_curve
from report import start_run, log_epoch, end_training


def train():

    # ── Timer globale ────────────────────────────────────────────────────────
    start_time = time_start()

    # ── Config ───────────────────────────────────────────────────────────────
    config = load_config("configs/config.yaml")

    # Avvia il file di log
    start_run(config)

    # Seed per riproducibilità
    set_seed(42)

    device = get_device()
    print(f"Training on: {device}")

    # ── Dataset e DataLoader ─────────────────────────────────────────────────
    train_dataset = MusicNetPianoDataset(
        csv_file=config['dataset']['csv_file'],
        data_dir=config['dataset']['data_dir'],
        split='train',
        chunk_duration=config['dataset']['chunk_duration'],
        sample_rate=config['dataset']['sample_rate'],
    )
    
    print(f"Tracce di training trovate: {len(train_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
    )

    # ── Modello ──────────────────────────────────────────────────────────────
    model = PianoTranscriptArchitecture(
        input_features=config['model']['input_features'],
        dropout=config['model']['dropout'],
    ).to(device)

    # ── Loss ─────────────────────────────────────────────────────────────────
    # BCEWithLogitsLoss per classificazione multi-label.
    # pos_weight > 1 penalizza di più le note mancate (falsi negativi),
    # utile perché le note attive sono molto meno frequenti dei silenzi.
    pos_weight = torch.ones(84).to(device) * config['training']['pos_weight']
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ── Ottimizzatore e scheduler ─────────────────────────────────────────────
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
    )

    # Dimezza il LR se la loss non migliora per N epoche
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=config['training']['scheduler_patience'],
        factor=config['training']['scheduler_factor'],
    )

    epochs     = config['training']['epochs']
    best_loss  = float('inf')
    train_losses = []

    # ── Loop di training ──────────────────────────────────────────────────────
    model.train()

    for epoch in range(epochs):
        start_time_epoch = time_start()
        epoch_loss = 0.0

        for batch in train_loader:

            waveforms = batch["waveform"]
            labels    = batch["labels"].to(device)   # (B, T, 84)

            # Estrazione CQT per ogni audio nel batch
            cqt_list = [extract_cqt(wave) for wave in waveforms]
            inputs   = torch.stack(cqt_list).to(device)  # (B, T, 84)

            # Azzera i gradienti
            optimizer.zero_grad()

            # Forward pass → (B, T, 84) logit
            outputs = model(inputs)

            # Allineamento temporale (CQT e piano-roll possono differire di 1 frame)
            min_frames = min(outputs.size(1), labels.size(1))
            outputs    = outputs[:, :min_frames, :]
            labels     = labels[:, :min_frames, :]

            # Loss + backprop
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)

        scheduler.step(avg_loss)

        # Salva il checkpoint se la loss migliora
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(
                {
                    'state_dict': model.state_dict(),
                    'optimizer':  optimizer.state_dict(),
                    'epoch':      epoch + 1,
                    'loss':       best_loss,
                },
                filename=config['training']['checkpoint_path'],
            )
            print(f"  → Nuovo best model salvato (loss: {best_loss:.4f})")

        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time_stop(start_time_epoch)
        print(f"Epoch {epoch+1}/{epochs}  |  Loss: {avg_loss:.4f}  |  LR: {current_lr:.6f}")
        log_epoch(epoch, avg_loss, current_lr, epoch_time)

    # ── Fine training ─────────────────────────────────────────────────────────
    print_time(time_stop(start_time))
    end_training()

    # Salva la curva di loss
    np.save('checkpoints/train_losses.npy', np.array(train_losses))
    print("Train losses salvate in checkpoints/train_losses.npy")
    plot_loss_curve(train_losses)


if __name__ == "__main__":
    train()