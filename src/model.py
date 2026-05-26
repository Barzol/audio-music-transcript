# model.py  –  Phase 3: CNN + BiLSTM + Multi-Output
#
# Differenze rispetto alla Phase 2 (CNN pura):
#   - Aggiunto layer BiLSTM tra CNN backbone e le 3 teste
#   - Il BiLSTM processa la sequenza temporale di feature CNN (B, T, 1344)
#     e produce (B, T, hidden_size * 2) grazie alla bidirezionalità
#   - Le 3 teste ricevono hidden_size*2 invece di 1344
#   - Parametri aggiuntivi: hidden_size, lstm_layers
#
# Flusso tensori:
#   (B, T, 84) CQT
#       ↓  unsqueeze
#   (B, 1, T, 84)
#       ↓  block1
#   (B, 32, T, 42)
#       ↓  block2
#   (B, 64, T, 21)
#       ↓  permute + view
#   (B, T, 1344)   ← feature CNN per frame
#       ↓  BiLSTM
#   (B, T, hidden_size*2)  ← contesto temporale bidirezionale
#       ↓          ↓          ↓
#   head_pitch  head_onset  head_offset
#       ↓          ↓          ↓
#   (B, T, 84)  (B, T, 84)  (B, T, 84)

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
        input_features: int = 84,    # bin CQT / note MIDI
        dropout: float = 0.3,
        hidden_size: int = 256,      # unità per direzione del BiLSTM
        lstm_layers: int = 2,        # strati BiLSTM impilati
    ):
        super(PianoTranscriptArchitecture, self).__init__()

        # ── CNN Backbone ──────────────────────────────────────────────────────
        # Identico alla Phase 2: estrae feature frequenziali dal CQT

        # Block 1: (B, 1, T, 84) → (B, 32, T, 42)
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
        )

        # Block 2: (B, 32, T, 42) → (B, 64, T, 21)
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool2d(kernel_size=(1, 2)),
        )

        cnn_out_dim = 64 * (input_features // 4)   # 64 * 21 = 1344

        # ── BiLSTM ────────────────────────────────────────────────────────────
        # Processa la sequenza temporale di feature CNN in entrambe le direzioni.
        # Dropout inter-layer attivo solo se lstm_layers > 1.
        self.bilstm = nn.LSTM(
            input_size=cnn_out_dim,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        lstm_out_dim = hidden_size * 2   # bidirezionale → output doppio

        # ── Tre teste parallele ───────────────────────────────────────────────
        # Ogni testa riceve le feature BiLSTM e produce 84 logit per frame.
        # FC leggero: il BiLSTM ha già estratto il contesto temporale.
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

    # ── Forward ───────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor):
        """
        x : (batch, time_frames, 84)
        returns : (logit_pitch, logit_onset, logit_offset)  ciascuno (B, T, 84)
        """
        # (B, T, 84) → (B, 1, T, 84)
        x = x.unsqueeze(1)

        # CNN backbone
        x = self.block1(x)   # (B, 32, T, 42)
        x = self.block2(x)   # (B, 64, T, 21)

        # Reshape per il BiLSTM: (B, 64, T, 21) → (B, T, 1344)
        B, C, T, F = x.size()
        x = x.permute(0, 2, 1, 3).contiguous().view(B, T, C * F)

        # BiLSTM: (B, T, 1344) → (B, T, hidden_size*2)
        x, _ = self.bilstm(x)

        # Tre teste
        logit_pitch  = self.head_pitch(x)    # (B, T, 84)
        logit_onset  = self.head_onset(x)    # (B, T, 84)
        logit_offset = self.head_offset(x)   # (B, T, 84)

        return logit_pitch, logit_onset, logit_offset


# ── Test rapido ───────────────────────────────────────────────────────────────
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
    print(f"Pitch  : {pitch.shape}")    # atteso (4, 215, 84)
    print(f"Onset  : {onset.shape}")    # atteso (4, 215, 84)
    print(f"Offset : {offset.shape}")   # atteso (4, 215, 84)

    assert pitch.shape == onset.shape == offset.shape == dummy.shape
    print("Test passato.")

    # Parametri totali
    total = sum(p.numel() for p in model.parameters())
    print(f"Parametri totali: {total:,}")