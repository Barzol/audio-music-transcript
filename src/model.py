# This file defines the model class PianoTranscriptArchitecture
# defines the nn architecture
# the model receives a CQT spectogram and outputs per-frame
# note probabilities.

# Architecture :
#   CNN     : extracts frequency patterns from CQT
#   Head : a small MLP fully connected that maps the 84 classes

import torch
import torch.nn as nn


class PianoTranscriptArchitecture(nn.Module):
    """
    Input  : FloatTensor  (batch, time_frames, 84)    CQT spectrogram
    Output : FloatTensor  (batch, time_frames, 84)    logit per note per frame
    """

    def __init__(
        self,
        input_features: int = 84,   # bin CQT / MIDI notes
        dropout: float = 0.3,
    ):
        super(PianoTranscriptArchitecture, self).__init__()

        # --- Block 1 ---
        # (B, 1, T, 84) → (B, 32, T, 42)
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32,
                      kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),   # dimezza l'asse frequenze
        )

        # --- Block 2 ---
        # (B, 32, T, 42) → (B, 64, T, 21)
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool2d(kernel_size=(1, 2)),   # dimezza di nuovo l'asse frequenze
        )

        # features after CNN:  64 channels × (84 // 4) bin = 64 × 21 = 1344
        cnn_out_dim = 64 * (input_features // 4)
        
        
        # ── Tre teste parallele ───────────────────────────────────────────────
        # Ogni testa riceve le stesse feature CNN e produce 84 logit per frame
        def make_head():
            return nn.Sequential(
                nn.Linear(cnn_out_dim, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, 84),
            )
            
        self.head_pitch  = make_head()   # nota attiva
        self.head_onset  = make_head()   # inizio nota
        self.head_offset = make_head()   # fine nota

    # ── Forward ───────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, time_frames, 84)
        returns logits : (batch, time_frames, 84)
        """

        # Aggiunge la dimensione del canale per Conv2d
        # (B, T, 84) → (B, 1, T, 84)
        x = x.unsqueeze(1)

        # Passaggi CNN
        x = self.block1(x)   # → (B, 32, T, 42)
        x = self.block2(x)   # → (B, 64, T, 21)

        # Riorganizza per la testa lineare
        # (B, 64, T, 21) → (B, T, 64, 21) → (B, T, 1344)
        B, C, T, F = x.size()
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(B, T, C * F)

        # Classificazione frame-by-frame
        # (B, T, 1344) → (B, T, 84)
        logit_pitch  = self.head_pitch(x)    # (B, T, 84)
        logit_onset  = self.head_onset(x)    # (B, T, 84)
        logit_offset = self.head_offset(x)   # (B, T, 84)

        return logit_pitch, logit_onset, logit_offset


# ── Test rapido ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = PianoTranscriptArchitecture()
    dummy = torch.randn(8, 215, 84)
    pitch, onset, offset = model(dummy)
 
    print(f"Input  : {dummy.shape}")
    print(f"Pitch  : {pitch.shape}")    # atteso (8, 215, 84)
    print(f"Onset  : {onset.shape}")    # atteso (8, 215, 84)
    print(f"Offset : {offset.shape}")   # atteso (8, 215, 84)
 
    assert pitch.shape == onset.shape == offset.shape == dummy.shape
    print("Test passato.")
 
"""
Flusso dei tensori
──────────────────
(B, T, 84)            input CQT
    ↓  unsqueeze
(B, 1, T, 84)
    ↓  block1
(B, 32, T, 42)
    ↓  block2
(B, 64, T, 21)
    ↓  permute + view
(B, T, 1344)          feature condivise
    ↓              ↓              ↓
head_pitch     head_onset     head_offset
    ↓              ↓              ↓
(B, T, 84)     (B, T, 84)     (B, T, 84)
"""