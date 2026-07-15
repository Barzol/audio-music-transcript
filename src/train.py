
import torch
import torch.nn as nn
import torch.optim as optim 
from torch.utils.data import DataLoader
from dataset import MusicNetPianoDataset
from model import PianoTranscriptArchitecture
from utils import extract_cqt, get_device, set_seed, save_checkpoint, load_config, time_start, time_stop, print_time

import numpy as np
from plots import plot_loss_curve
from report import start_run, log_epoch, end_training

def train():

    start_time = time_start()

    config = load_config("configs/config.yaml")

    start_run(config)

    set_seed(42)

    device = get_device()
    print(f"Training on : {device}")

    train_dataset = MusicNetPianoDataset(
        csv_file = config['dataset']['csv_file'],
        data_dir = config['dataset']['data_dir'],
        split='train',
        chunk_duration = config['dataset']['chunk_duration'],
        sample_rate = config['dataset']['sample_rate']
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size = config['training']['batch_size'], 
        shuffle = True
    )

    model = PianoTranscriptArchitecture(
        input_features = config['model']['input_features'],
        hidden_size = config['model']['hidden_size'],
        lstm_layers = config['model']['lstm_layers'],
        dropout = config['model']['dropout']
    ).to(device)


    pos_weight = torch.ones(84).to(device) * config['training']['pos_weight']

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.Adam(
        model.parameters(), 
        lr = config['training']['learning_rate']
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode = 'min',
        patience = config['training']['scheduler_patience'],
        factor = config['training']['scheduler_factor']
    )

    epochs = config['training']['epochs']
    best_loss = float('inf')
    train_losses = []

    model.train()

    for epoch in range(epochs):
        start_time_epoch = time_start();
        epoch_loss = 0.0

        for batch in train_loader :

            waveforms = batch["waveform"]
            labels = batch["labels"].to(device)

            cqt_list = [extract_cqt(wave) for wave in waveforms]

            inputs = torch.stack(cqt_list).to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            min_frames = min(outputs.size(1), labels.size(1))
            outputs = outputs[:, :min_frames, :]
            labels = labels[:, :min_frames, :]

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss/len(train_loader)

        train_losses.append(avg_loss)

        scheduler.step(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint({
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1,
                'loss' : best_loss
            }, filename = config['training']['checkpoint_path'])
            
            print(f"New best model saved (loss: {best_loss:.4f})")

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs} - Loss : {avg_loss:.4f}")

        log_epoch(epoch, avg_loss, current_lr, time_stop(start_time=start_time_epoch))

    print_time(time_stop(start_time))

    end_training()

    np.save('checkpoints/train_losses.npy', np.array(train_losses))
    print("Train losses saved to checkpoints/train_losses.npy")
    plot_loss_curve(train_losses)

    save_checkpoint({
        'state_dict' : model.state_dict(),
        'optimizer' : optimizer.state_dict()
    }, filename="best_model.pt")

    

if __name__ == "__main__":
    train()