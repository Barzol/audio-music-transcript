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

        # --- Head ---
        self.head = nn.Sequential(
            nn.Linear(cnn_out_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 84),
        )

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
        logits = self.head(x)

        return logits


# ── Test rapido ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = BaselineCNN()
    dummy = torch.randn(8, 215, 84)   # batch=8, 215 frame CQT, 84 bin
    out   = model(dummy)
    print(f"Input  shape : {dummy.shape}")
    print(f"Output shape : {out.shape}")   # atteso: [8, 215, 84]
    assert out.shape == dummy.shape, "Shape mismatch!"
    print("Test passato.")

"""
Flusso dei tensori
──────────────────
(8, 215, 84)          input CQT
    ↓  unsqueeze
(8, 1, 215, 84)       aggiunge il canale
    ↓  block1 (Conv + BN + ReLU + MaxPool)
(8, 32, 215, 42)      feature frequenziali, metà bin
    ↓  block2 (Conv + BN + ReLU + Dropout + MaxPool)
(8, 64, 215, 21)      feature più astratte, un quarto dei bin
    ↓  permute + view
(8, 215, 1344)        flatten per frame
    ↓  head (Linear → ReLU → Dropout → Linear)
(8, 215, 84)          logit per nota per frame
"""