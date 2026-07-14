# Phase 2 MAESTRO — training loop
# Multi-output: pitch + onset + offset with weighted BCE loss

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import MaestroDataset
from model   import PianoTranscriptArchitecture
from utils   import extract_cqt, get_device, set_seed, save_checkpoint, load_config, \
                    time_start, time_stop, print_time

import numpy as np
from plots  import plot_loss_curve
from report import start_run, log_epoch, end_training


def train():

    start_time = time_start()
    config     = load_config()
    start_run(config)
    set_seed(42)

    device = get_device()
    print(f"Training on: {device}")

    # ── Datasets ───────────────────────────────────────────────────────────
    train_dataset = MaestroDataset(split='train')
    val_dataset   = MaestroDataset(split='validation')

    train_loader = DataLoader(train_dataset,
                              batch_size=config['training']['batch_size'],
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset,
                              batch_size=config['training']['batch_size'],
                              shuffle=False, num_workers=4, pin_memory=True)

    # ── Model ──────────────────────────────────────────────────────────────
    model = PianoTranscriptArchitecture(
        input_features=config['model']['input_features'],
        dropout=config['model']['dropout']
    ).to(device)

    # ── Loss ───────────────────────────────────────────────────────────────
    pw_pitch  = torch.full((84,), config['training']['pos_weight']).to(device)
    pw_onset  = torch.full((84,), config['training']['pos_weight_onset']).to(device)
    pw_offset = torch.full((84,), config['training']['pos_weight_offset']).to(device)

    crit_pitch  = nn.BCEWithLogitsLoss(pos_weight=pw_pitch)
    crit_onset  = nn.BCEWithLogitsLoss(pos_weight=pw_onset)
    crit_offset = nn.BCEWithLogitsLoss(pos_weight=pw_offset)

    lw_onset  = config['training']['loss_weight_onset']
    lw_offset = config['training']['loss_weight_offset']

    # ── Optimizer & Scheduler ──────────────────────────────────────────────
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min',
        patience=config['training']['scheduler_patience'],
        factor=config['training']['scheduler_factor']
    )

    epochs      = config['training']['epochs']
    best_loss   = float('inf')
    train_losses, val_losses = [], []

    # ── Training loop ──────────────────────────────────────────────────────
    for epoch in range(epochs):
        t0         = time_start()
        epoch_loss = 0.0

        model.train()
        for batch in train_loader:
            waveforms      = batch['waveform']
            labels         = batch['labels'].to(device)
            onset_labels   = batch['onset_labels'].to(device)
            offset_labels  = batch['offset_labels'].to(device)

            cqt_list = []
            for wave in waveforms:
                c = extract_cqt(wave.squeeze(), hop_length=config['dataset']['hop_length'])
                cqt_list.append(c)
            inputs = torch.stack(cqt_list).to(device)

            optimizer.zero_grad()
            out_pitch, out_onset, out_offset = model(inputs)

            T = min(out_pitch.size(1), labels.size(1))
            out_pitch  = out_pitch[:, :T, :]
            out_onset  = out_onset[:, :T, :]
            out_offset = out_offset[:, :T, :]
            labels        = labels[:, :T, :]
            onset_labels  = onset_labels[:, :T, :]
            offset_labels = offset_labels[:, :T, :]

            loss_pitch  = crit_pitch(out_pitch,  labels)
            loss_onset  = crit_onset(out_onset,  onset_labels)
            loss_offset = crit_offset(out_offset, offset_labels)
            loss = loss_pitch + lw_onset * loss_onset + lw_offset * loss_offset

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_train = epoch_loss / len(train_loader)
        train_losses.append(avg_train)

        # ── Validation ─────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                waveforms      = batch['waveform']
                labels         = batch['labels'].to(device)
                onset_labels   = batch['onset_labels'].to(device)
                offset_labels  = batch['offset_labels'].to(device)

                cqt_list = []
                for wave in waveforms:
                    c = extract_cqt(wave.squeeze(), hop_length=config['dataset']['hop_length'])
                    cqt_list.append(c)
                inputs = torch.stack(cqt_list).to(device)

                out_pitch, out_onset, out_offset = model(inputs)

                T = min(out_pitch.size(1), labels.size(1))
                out_pitch  = out_pitch[:, :T, :]
                out_onset  = out_onset[:, :T, :]
                out_offset = out_offset[:, :T, :]
                labels        = labels[:, :T, :]
                onset_labels  = onset_labels[:, :T, :]
                offset_labels = offset_labels[:, :T, :]

                loss_pitch  = crit_pitch(out_pitch,  labels)
                loss_onset  = crit_onset(out_onset,  onset_labels)
                loss_offset = crit_offset(out_offset, offset_labels)
                val_loss += (loss_pitch + lw_onset * loss_onset + lw_offset * loss_offset).item()

        avg_val = val_loss / len(val_loader)
        val_losses.append(avg_val)
        scheduler.step(avg_val)

        # Early stopping on min LR
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr < config['training']['min_lr']:
            print(f"LR {current_lr:.6f} below minimum. Early stopping.")
            break

        if avg_val < best_loss:
            best_loss = avg_val
            save_checkpoint({
                'state_dict': model.state_dict(),
                'optimizer':  optimizer.state_dict(),
                'epoch':      epoch + 1,
                'loss':       best_loss
            }, filename=config['training']['checkpoint_path'])
            print(f"Best model saved (val loss: {best_loss:.4f})")

        duration = time_stop(t0)
        print(f"Epoch {epoch+1}/{epochs} — Train: {avg_train:.4f}  Val: {avg_val:.4f}  LR: {current_lr:.6f}")
        log_epoch(epoch, avg_train, avg_val, current_lr, duration)

    print_time(time_stop(start_time))
    end_training()

    np.save('checkpoints/train_losses.npy', np.array(train_losses))
    np.save('checkpoints/val_losses.npy',   np.array(val_losses))
    plot_loss_curve(train_losses, val_losses)
    save_checkpoint({'state_dict': model.state_dict(),
                     'optimizer':  optimizer.state_dict()},
                    filename='final_model.pt')


if __name__ == '__main__':
    train()