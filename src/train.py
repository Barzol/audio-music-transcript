# this file contains the main training loop 
# it loads the dataset, initializes the model, and trains it
# for a number of epochs

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

    # starts timer for training
    start_time = time_start()

    # load hyperparameters from the config file
    config = load_config()

    # starts log file
    start_run(config)

    # set random seed
    set_seed(42)

    device = get_device()
    print(f"Training on : {device}")

    # -------- Dataset and Dataloader --------
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
        num_workers = 0,
        pin_memory  = True
    )
 
    val_loader = DataLoader(
        val_dataset,
        batch_size  = config['training']['batch_size'],
        shuffle     = False,
        num_workers = 0,
        pin_memory  = True
    )

    # -------- Model -------------------------
    # CRNN model initialize
    model = PianoTranscriptArchitecture(
        input_features = config['model']['input_features'],
        dropout        = config['model']['dropout']
    ).to(device)
 

    # -------- Loss --------------------------
    # Loss : Binary Cross Entropy for multi-label classification
    # BCEWithLogitsLoss because the final sigmoid will be applied 
    # in post processing

    # this term tells the loss to penalize missing note 
    pos_weight = torch.full((84,), config['training']['pos_weight']).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # -------- Optimizer ---------------------
    # optimizer Adam with learning rate 0.001
    optimizer = optim.Adam(
        model.parameters(),
        lr = config['training']['learning_rate']
    )

    # reduces LR by 0.5 if loss doesnt' primove for 5 epochs
    # this helps escape plateaus 
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

    # -------- Training loop -----------------
    for epoch in range(epochs):
        start_time_epoch = time_start();
        epoch_loss = 0.0
        
        # --- Training ---
        model.train()
 
        for batch in train_loader:
 
            waveforms = batch["waveform"]
            labels    = batch["labels"].to(device)
 
            # CQT extraction for the batch
            cqt_list = []
            for wave in waveforms:
                c_feat = extract_cqt(wave.squeeze(), hop_length=config['dataset']['hop_length'])
                if isinstance(c_feat, np.ndarray):
                    c_feat = torch.from_numpy(c_feat)
                cqt_list.append(c_feat)
 
            inputs = torch.stack(cqt_list).to(device)
 
            optimizer.zero_grad()

            # forward pass : active notes per frames
            # output : (batch, time_frames, 84)
            outputs = model(inputs)

            # Align the temporal dimensions
            min_frames = min(outputs.size(1), labels.size(1))
            outputs    = outputs[:, :min_frames, :]
            labels     = labels[:, :min_frames, :]
 
            loss = criterion(outputs, labels)
            loss.backward()
            
            # gradient clipping : avoid exploding gradients in LSTMs
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
 
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
 
        
        # --- Validation ---
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
        
        # Early stopping on minimum LR
        min_lr     = config['training']['min_lr']
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr < min_lr:
            print(f"LR {current_lr:.6f} below minimum {min_lr:.6f}. Early stopping.")
            break
 
        # Save best checkpoint
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

        # log of epochs
        log_epoch(epoch, avg_train_loss, avg_val_loss, current_lr, duration)
 
    # end of training
    print_time(time_stop(start_time))
    end_training()
 
    np.save('checkpoints/train_losses.npy', np.array(train_losses))
    np.save('checkpoints/val_losses.npy',   np.array(val_losses))
    print("Losses saved to checkpoints/")
 
    plot_loss_curve(train_losses, val_losses)
 
    # save final weights
    save_checkpoint({
        'state_dict': model.state_dict(),
        'optimizer':  optimizer.state_dict()
    }, filename="final_model.pt")

    

if __name__ == "__main__":
    train()