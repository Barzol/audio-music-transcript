
import torch
import torch.nn as nn


class PianoTranscriptArchitecture(nn.Module):
    """
    Input  : FloatTensor  (batch, time_frames, 84)   CQT spectrogram
    Output : tuple di tre FloatTensor (batch, time_frames, 84)  logit per nota
             (logit_pitch, logit_onset, logit_offset)
    """

    def __init__(
        self,
        input_features: int = 84,
        dropout: float = 0.3,
        hidden_size: int = 256,
        lstm_layers: int = 2,
    ):
        super(PianoTranscriptArchitecture, self).__init__()


        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool2d(kernel_size=(1, 2)),
        )

        cnn_out_dim = 64 * (input_features // 4)

        self.bilstm = nn.LSTM(
            input_size=cnn_out_dim,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        lstm_out_dim = hidden_size * 2

        def make_head():
            return nn.Sequential(
                nn.Linear(lstm_out_dim, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 84),
            )

        self.head_pitch  = make_head()
        self.head_onset  = make_head()
        self.head_offset = make_head()

    def forward(self, x: torch.Tensor):
        """
        x : (batch, time_frames, 84)
        returns : (logit_pitch, logit_onset, logit_offset)  ciascuno (B, T, 84)
        """
        x = x.unsqueeze(1)

        x = self.block1(x)
        x = self.block2(x)

        B, C, T, F = x.size()
        x = x.permute(0, 2, 1, 3).contiguous().view(B, T, C * F)

        x, _ = self.bilstm(x)

        logit_pitch  = self.head_pitch(x)
        logit_onset  = self.head_onset(x)
        logit_offset = self.head_offset(x)

        return logit_pitch, logit_onset, logit_offset


if __name__ == "__main__":
    model = PianoTranscriptArchitecture(
        input_features=84,
        dropout=0.4,
        hidden_size=256,
        lstm_layers=2,
    )

    dummy = torch.randn(4, 215, 84)
    pitch, onset, offset = model(dummy)

    print(f"Input  : {dummy.shape}")
    print(f"Pitch  : {pitch.shape}")
    print(f"Onset  : {onset.shape}")
    print(f"Offset : {offset.shape}")

    assert pitch.shape == onset.shape == offset.shape == dummy.shape
    print("Test passato.")

    total = sum(p.numel() for p in model.parameters())
    print(f"Parametri totali: {total:,}")