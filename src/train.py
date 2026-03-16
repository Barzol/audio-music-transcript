import torch
import torch.nn as nn
import torch.optim as optim 
from torch.utils.data import DataLoader
from dataset import MusicNetPianoDataset
from model import PianoTranscriptArchitecture
from utils import extract_cqt, get_device, set_seed, save_checkpoint

def train():
    set_seed(42)
    device = get_device
    print("Training on : {device}")

    # Dataset and Dataloader with Batch Size 8
    train_dataset = MusicNetPianoDataset(split='train')
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    # CRNN model initialize
    model = PianoTranscriptArchitecture().to(device)

    # Loss : Binary Cross Entropy for multi-label classification
    # BCEWithLogitsLoss because the final sigmoid will be applied in post processing
    criterion = nn.BCEWithLogitsLoss()

    # optimizer Adam with learning rate 0.001
    optimizer = optim.Adam(model.parameters(), lr = 0.001)

    epochs = 10

    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0

        for batch in train_loader :
            waveforms = batch["waveform"]
            labels = batch["labels"].to(device)

            # extracting CQT features for batch
            cqt_list = [extract_cqt(wave) for wave in waveforms]
            inputs = torch.stack(cqt_list).to(device)

            optimizer.zero_grad()

            # forward pass : active notes per frames
            outputs = model(inputs)

            # Align the temporal dimensions
            min_frames = min(outputs.size(1), labels.size(1))
            outputs = outputs[:, :min_frames, :]
            labels = labels[:, :min_frames, :]

            # Calculate Loss and Backpropagation
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        
        print(f"Epoc {epoch+1}/{epochs} - Loss : {epoch_loss/len(train_loader):.4f}")

    save_checkpoint({
        'state_dict' : model.state_dict(),
        'optimizer' : optimizer.state_dict()
    }, filename="best_model.pt")

if __name__ == "__main__":
    train()