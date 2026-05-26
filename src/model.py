# This file defines the model class PianoTranscriptArchitecture
# defines the nn architecture
# the model receives a CQT spectogram and outputs per-frame
# note probabilities.

# Architecture :
#   CNN     : extracts frequency patterns from CQT
#   BiLSTM  : captures note onsets and offsets
#   Linear  : maps to 84 probabilities ( note probabilities )

import torch
import torch.nn as nn
from utils import load_config

class PianoTranscriptArchitecture(nn.Module):

    def __init__(self, input_features=84, dropout=0.4):
        super().__init__()
 
        # Shared CNN backbone (identical to Phase 1 BaselineCNN)
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2))        # (B, 32, T, 42)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool2d(kernel_size=(1, 2))        # (B, 64, T, 21)
        )
 
        # After two MaxPool(1,2): freq_bins = input_features // 4 = 21
        cnn_out_dim = 64 * (input_features // 4)   # 1344
 
        # Shared intermediate projection
        self.shared_fc = nn.Sequential(
            nn.Linear(cnn_out_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
 
        # Three independent heads
        self.head_pitch  = nn.Linear(256, input_features)
        self.head_onset  = nn.Linear(256, input_features)
        self.head_offset = nn.Linear(256, input_features)
 
    def forward(self, x):
        # x : (B, T, 84)
        x = x.unsqueeze(1)                         # (B, 1, T, 84)
        x = self.block1(x)                         # (B, 32, T, 42)
        x = self.block2(x)                         # (B, 64, T, 21)
 
        B, C, T, F = x.size()
        x = x.permute(0, 2, 1, 3).contiguous()    # (B, T, 64, 21)
        x = x.view(B, T, C * F)                   # (B, T, 1344)
 
        shared = self.shared_fc(x)                 # (B, T, 256)
 
        pitch  = self.head_pitch(shared)            # (B, T, 84)
        onset  = self.head_onset(shared)            # (B, T, 84)
        offset = self.head_offset(shared)           # (B, T, 84)
 
        return pitch, onset, offset

# --- TEST DEL MODELLO ---
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