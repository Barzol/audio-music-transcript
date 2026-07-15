
import torch
import torch.nn as nn
import torch.optim as optim 
from torch.utils.data import DataLoader

from dataset import MaestroDataset
from model import PianoTranscriptArchitecture
from utils import extract_cqt, get_device, set_seed, save_checkpoint, load_config, time_start, time_stop, print_time, load_checkpoint

import numpy as np
from plots import plot_loss_curve
from report import start_run, log_epoch, end_training

def train():

    start_time = time_start()

    config = load_config()

    start_run(config)

    set_seed(42)

    device = get_device()
    print(f"Training on : {device}")

    train_dataset = MaestroDataset(
        csv_file       = config['dataset']['csv_file'],
        data_dir       = config['dataset']['data_dir'],
        split          = 'train',
        chunk_duration = config['dataset']['chunk_duration'],
        sample_rate    = config['dataset']['sample_rate']
    )
 
    val_dataset = MaestroDataset(
        csv_file       = config['dataset']['csv_file'],
        data_dir       = config['dataset']['data_dir'],
        split          = 'validation',
        chunk_duration = config['dataset']['chunk_duration'],
        sample_rate    = config['dataset']['sample_rate']
    )
 
    train_loader = DataLoader(
        train_dataset,
        batch_size  = config['training']['batch_size'],
        shuffle     = True,
        num_workers = 4,
        pin_memory  = True,
        persistent_workers = True
    )
 
    val_loader = DataLoader(
        val_dataset,
        batch_size  = config['training']['batch_size'],
        shuffle     = False,
        num_workers = 4,
        pin_memory  = True,
        persistent_workers = True
    )

    model = PianoTranscriptArchitecture(
        input_features = config['model']['input_features'],
        dropout        = config['model']['dropout']
    ).to(device)
 


    pos_weight = torch.full((84,), config['training']['pos_weight']).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.Adam(
        model.parameters(),
        lr = config['training']['learning_rate']
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode      = 'min',
        patience  = config['training']['scheduler_patience'],
        factor    = config['training']['scheduler_factor']
    )
 
    epochs      = config['training']['epochs']
    best_loss   = float('inf')
    train_losses = []
    val_losses   = []

    for epoch in range(epochs):
        start_time_epoch = time_start();
        epoch_loss = 0.0
        
        model.train()
 
        for batch in train_loader:
 
            waveforms = batch["waveform"]
            labels    = batch["labels"].to(device)
 
            cqt_list = []
            for wave in waveforms:
                c_feat = extract_cqt(wave.squeeze(), hop_length=config['dataset']['hop_length'])
                if isinstance(c_feat, np.ndarray):
                    c_feat = torch.from_numpy(c_feat)
                cqt_list.append(c_feat)
 
            inputs = torch.stack(cqt_list).to(device)
 
            optimizer.zero_grad()

            outputs = model(inputs)

            min_frames = min(outputs.size(1), labels.size(1))
            outputs    = outputs[:, :min_frames, :]
            labels     = labels[:, :min_frames, :]
 
            loss = criterion(outputs, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
 
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
 
        
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                waveforms = batch["waveform"]
                labels    = batch["labels"].to(device)
 
                cqt_list = []
                for wave in waveforms:
                    c_feat = extract_cqt(wave.squeeze(), hop_length=config['dataset']['hop_length'])
                    if isinstance(c_feat, np.ndarray):
                        c_feat = torch.from_numpy(c_feat)
                    cqt_list.append(c_feat)
 
                inputs  = torch.stack(cqt_list).to(device)
                outputs = model(inputs)
 
                min_frames = min(outputs.size(1), labels.size(1))
                outputs    = outputs[:, :min_frames, :]
                labels     = labels[:, :min_frames, :]
 
                loss      = criterion(outputs, labels)
                val_loss += loss.item()
            
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
 
        scheduler.step(avg_val_loss)
        
        min_lr     = config['training']['min_lr']
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr < min_lr:
            print(f"LR {current_lr:.6f} below minimum {min_lr:.6f}. Early stopping.")
            break
 
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            save_checkpoint({
                'state_dict': model.state_dict(),
                'optimizer':  optimizer.state_dict(),
                'epoch':      epoch + 1,
                'loss':       best_loss
            }, filename=config['training']['checkpoint_path'])
            print(f"New best model saved (val loss: {best_loss:.4f})")
 
        duration   = time_stop(start_time=start_time_epoch)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs} — Train: {avg_train_loss:.4f}  Val: {avg_val_loss:.4f}  LR: {current_lr:.6f}")

        log_epoch(epoch, avg_train_loss, avg_val_loss, current_lr, duration)
 
    print_time(time_stop(start_time))
    end_training()
 
    np.save('checkpoints/train_losses.npy', np.array(train_losses))
    np.save('checkpoints/val_losses.npy',   np.array(val_losses))
    print("Losses saved to checkpoints/")
 
    plot_loss_curve(train_losses, val_losses)
 
    save_checkpoint({
        'state_dict': model.state_dict(),
        'optimizer':  optimizer.state_dict()
    }, filename="final_model.pt")

    

if __name__ == "__main__":
    train()