

import torch
import torch.nn as nn
from utils import load_config

class PianoTranscriptArchitecture(nn.Module):

    def __init__(self, input_features=84, dropout=0.4):
        super().__init__()
 
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2))
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool2d(kernel_size=(1, 2))
        )
 
        cnn_out_dim = 64 * (input_features // 4)
 
        self.shared_fc = nn.Sequential(
            nn.Linear(cnn_out_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
 
        self.head_pitch  = nn.Linear(256, input_features)
        self.head_onset  = nn.Linear(256, input_features)
        self.head_offset = nn.Linear(256, input_features)
 
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
 
        B, C, T, F = x.size()
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(B, T, C * F)
 
        shared = self.shared_fc(x)
 
        pitch  = self.head_pitch(shared)
        onset  = self.head_onset(shared)
        offset = self.head_offset(shared)
 
        return pitch, onset, offset

if __name__ == "__main__":
    model = PianoTranscriptArchitecture(
        input_features=84,
        hidden_size=256,
        lstm_layers=2,
        dropout=0.3
    )
    dummy_input = torch.randn(8, 215, 84)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")

'''
A tiny scheme

(8, 215, 84)          input CQT
    ↓ unsqueeze
(8, 1, 215, 84)       add channel dimension
    ↓ CNN
(8, 64, 215, 21)      extracted features with frequencies halved 2 times
    ↓ permute + view
(8, 215, 1344)        flatten for LSTM
    ↓ BiLSTM
(8, 215, 256)         BiLSTM gives forward and backward temporal context
    ↓ Linear
(8, 215, 84)          per-frame note probabilities


'''