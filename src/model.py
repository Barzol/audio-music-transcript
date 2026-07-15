

import torch
import torch.nn as nn


class PianoTranscriptArchitecture(nn.Module):
    """
    Input  : FloatTensor  (batch, time_frames, 84)    CQT spectrogram
    Output : FloatTensor  (batch, time_frames, 84)    logit per note per frame
    """

    def __init__(
        self,
        input_features: int = 84,
        dropout: float = 0.3,
    ):
        super(PianoTranscriptArchitecture, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32,
                      kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool2d(kernel_size=(1, 2)),
        )

        cnn_out_dim = 64 * (input_features // 4)

        self.head = nn.Sequential(
            nn.Linear(cnn_out_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 84),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, time_frames, 84)
        returns logits : (batch, time_frames, 84)
        """

        x = x.unsqueeze(1)

        x = self.block1(x)
        x = self.block2(x)

        B, C, T, F = x.size()
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(B, T, C * F)

        logits = self.head(x)

        return logits


if __name__ == "__main__":
    model = BaselineCNN()
    dummy = torch.randn(8, 215, 84)
    out   = model(dummy)
    print(f"Input  shape : {dummy.shape}")
    print(f"Output shape : {out.shape}")
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