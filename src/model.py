# model.py  –  Phase 3: CNN + BiLSTM + Multi-Output
#
# Architecture:
#   CNN backbone  : estrae feature frequenziali dal CQT/STFT/Mel
#   BiLSTM        : contesto temporale bidirezionale
#   3 teste       : pitch, onset, offset — ciascuna (B, T, num_notes)
#
# Parametri chiave:
#   input_features : bin di frequenza in input (84 CQT MusicNet, 88 CQT MAESTRO,
#                    ~193 STFT, 128 Mel)
#   num_notes      : note MIDI in output (84 MusicNet, 88 MAESTRO).
#                    Se None, usa input_features.
#   hidden_size    : unità per direzione del BiLSTM
#   lstm_layers    : strati BiLSTM impilati

import torch
import torch.nn as nn


class PianoTranscriptArchitecture(nn.Module):

    def __init__(
        self,
        input_features: int = 84,
        num_notes: int = None,
        dropout: float = 0.3,
        hidden_size: int = 128,
        lstm_layers: int = 1,
    ):
        super().__init__()

        if num_notes is None:
            num_notes = input_features
        self.num_notes = num_notes

        # ── CNN Backbone ──────────────────────────────────────────────────────
        # Block 1: (B, 1, T, F) → (B, 32, T, F//2)
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
        )

        # Block 2: (B, 32, T, F//2) → (B, 64, T, F//4)
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool2d(kernel_size=(1, 2)),
        )

        cnn_out_dim = 64 * (input_features // 4)

        # ── BiLSTM ────────────────────────────────────────────────────────────
        self.bilstm = nn.LSTM(
            input_size=cnn_out_dim,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        lstm_out_dim = hidden_size * 2

        # ── Tre teste parallele ───────────────────────────────────────────────
        def make_head():
            return nn.Sequential(
                nn.Linear(lstm_out_dim, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, num_notes),
            )

        self.head_pitch  = make_head()
        self.head_onset  = make_head()
        self.head_offset = make_head()

    def forward(self, x: torch.Tensor):
        """
        x : (B, T, input_features)
        returns : (logit_pitch, logit_onset, logit_offset)  ciascuno (B, T, num_notes)
        """
        x = x.unsqueeze(1)              # (B, 1, T, F)
        x = self.block1(x)              # (B, 32, T, F//2)
        x = self.block2(x)              # (B, 64, T, F//4)

        B, C, T, F = x.size()
        x = x.permute(0, 2, 1, 3).contiguous().view(B, T, C * F)  # (B, T, cnn_out_dim)

        x, _ = self.bilstm(x)          # (B, T, hidden_size*2)

        return self.head_pitch(x), self.head_onset(x), self.head_offset(x)


# ── Test ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # MusicNet: 84 bin CQT, 84 note
    m = PianoTranscriptArchitecture(input_features=84, num_notes=84,
                                    hidden_size=128, lstm_layers=1, dropout=0.3)
    d = torch.randn(4, 215, 84)
    p, on, off = m(d)
    assert p.shape == (4, 215, 84)
    print(f"MusicNet CQT  OK  {d.shape} → {p.shape}")

    # MAESTRO: 88 bin CQT, 88 note
    m = PianoTranscriptArchitecture(input_features=88, num_notes=88,
                                    hidden_size=128, lstm_layers=1, dropout=0.3)
    d = torch.randn(4, 215, 88)
    p, on, off = m(d)
    assert p.shape == (4, 215, 88)
    print(f"MAESTRO CQT   OK  {d.shape} → {p.shape}")

    # STFT MusicNet: 193 bin, 84 note
    m = PianoTranscriptArchitecture(input_features=193, num_notes=84,
                                    hidden_size=128, lstm_layers=1, dropout=0.3)
    d = torch.randn(4, 215, 193)
    p, on, off = m(d)
    assert p.shape == (4, 215, 84)
    print(f"MusicNet STFT OK  {d.shape} → {p.shape}")

    print("Tutti i test passati.")
    total = sum(p.numel() for p in m.parameters())
    print(f"Parametri totali (STFT config): {total:,}")