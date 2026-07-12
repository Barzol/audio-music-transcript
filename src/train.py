# train.py  –  Fase 2: Multi-Output CNN
#
# Differenze rispetto alla Fase 1:
#   - Il modello restituisce tre output: (logit_pitch, logit_onset, logit_offset)
#   - Tre loss separate con pos_weight distinti (onset/offset sono più rari)
#   - Loss totale = w_pitch*L_pitch + w_onset*L_onset + w_offset*L_offset
#     (pesi configurabili in configs/config.yaml)
#   - Il batch ora estrae anche 'onsets' e 'offsets' dal dataset
#   - Loop di validazione al termine di ogni epoch
#   - Lo scheduler usa val_loss invece di train_loss
#   - Il checkpoint viene salvato sul minimo di val_loss
 
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
 
 
def compute_loss(model_out, labels, onsets, offsets,
                 criterion_pitch, criterion_onset, criterion_offset,
                 w_pitch, w_onset, w_offset, device):
    """Calcola la loss totale pesata allineando le dimensioni temporali."""
    logit_pitch, logit_onset, logit_offset = model_out
 
    T = min(logit_pitch.size(1), labels.size(1))
    logit_pitch  = logit_pitch[:, :T, :]
    logit_onset  = logit_onset[:, :T, :]
    logit_offset = logit_offset[:, :T, :]
    labels  = labels[:, :T, :]
    onsets  = onsets[:, :T, :]
    offsets = offsets[:, :T, :]
 
    loss_pitch  = criterion_pitch(logit_pitch,  labels)
    loss_onset  = criterion_onset(logit_onset,  onsets)
    loss_offset = criterion_offset(logit_offset, offsets)
 
    return w_pitch * loss_pitch + w_onset * loss_onset + w_offset * loss_offset
 
 
def train():
 
    start_time = time_start()
 
    config = load_config("configs/config.yaml")
    start_run(config)
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
    val_dataset = MusicNetPianoDataset(
        csv_file=config['dataset']['csv_file'],
        data_dir=config['dataset']['data_dir'],
        split='val',
        chunk_duration=config['dataset']['chunk_duration'],
        sample_rate=config['dataset']['sample_rate'],
    )
 
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
 
    print(f"Tracce — train: {len(train_dataset)} | val: {len(val_dataset)}")
 
    # ── Modello ──────────────────────────────────────────────────────────────
    model = PianoTranscriptArchitecture(
        input_features=config['model']['input_features'],
        dropout=config['model']['dropout'],
    ).to(device)
 
    # ── Loss ─────────────────────────────────────────────────────────────────
    pw_pitch  = torch.ones(84).to(device) * config['training']['pos_weight_pitch']
    pw_onset  = torch.ones(84).to(device) * config['training']['pos_weight_onset']
    pw_offset = torch.ones(84).to(device) * config['training']['pos_weight_offset']
 
    criterion_pitch  = nn.BCEWithLogitsLoss(pos_weight=pw_pitch)
    criterion_onset  = nn.BCEWithLogitsLoss(pos_weight=pw_onset)
    criterion_offset = nn.BCEWithLogitsLoss(pos_weight=pw_offset)
 
    w_pitch  = config['training']['loss_weight_pitch']
    w_onset  = config['training']['loss_weight_onset']
    w_offset = config['training']['loss_weight_offset']
 
    # ── Ottimizzatore e scheduler ─────────────────────────────────────────────
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
 
    # Lo scheduler monitora la val_loss
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=config['training']['scheduler_patience'],
        factor=config['training']['scheduler_factor'],
    )
 
    epochs        = config['training']['epochs']
    best_val_loss = float('inf')
    train_losses  = []
    val_losses    = []
 
    # ── Loop di training ──────────────────────────────────────────────────────
    for epoch in range(epochs):
        start_time_epoch = time_start()
 
        # — Train —
        model.train()
        epoch_loss = 0.0
 
        for batch in train_loader:
            waveforms = batch["waveform"]
            labels    = batch["labels"].to(device)
            onsets    = batch["onsets"].to(device)
            offsets   = batch["offsets"].to(device)
 
            cqt_list = [extract_cqt(wave) for wave in waveforms]
            inputs   = torch.stack(cqt_list).to(device)
 
            optimizer.zero_grad()
            out  = model(inputs)
            loss = compute_loss(out, labels, onsets, offsets,
                                criterion_pitch, criterion_onset, criterion_offset,
                                w_pitch, w_onset, w_offset, device)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
 
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
 
        # — Validation —
        model.eval()
        val_loss = 0.0
 
        with torch.no_grad():
            for batch in val_loader:
                waveforms = batch["waveform"]
                labels    = batch["labels"].to(device)
                onsets    = batch["onsets"].to(device)
                offsets   = batch["offsets"].to(device)
 
                cqt_list = [extract_cqt(wave) for wave in waveforms]
                inputs   = torch.stack(cqt_list).to(device)
 
                out  = model(inputs)
                loss = compute_loss(out, labels, onsets, offsets,
                                    criterion_pitch, criterion_onset, criterion_offset,
                                    w_pitch, w_onset, w_offset, device)
                val_loss += loss.item()
 
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
 
        # Scheduler su val_loss
        scheduler.step(avg_val_loss)
 
        # Salva il checkpoint sul minimo di val_loss
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
            print(f"  → Nuovo best model salvato (val_loss: {best_val_loss:.4f})")
 
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time_stop(start_time_epoch)
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