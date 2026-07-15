

import torch
import torch.nn as nn

class PianoTranscriptArchitecture(nn.Module):

    def __init__(
            self,
            input_features = 84,
            hidden_size = 128,
            lstm_layers = 1,        
            dropout = 0.3
            ):
        

        super(PianoTranscriptArchitecture, self).__init__()


        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,2)),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool2d(kernel_size=(1,2)),            
        )

        cnn_out_freq = input_features // 4
        cnn_out_features = 64 * cnn_out_freq


        self.bilstm = nn.LSTM(
            input_size=cnn_out_features,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True
        )


        self.fc = nn.Linear(hidden_size * 2, 84)


    def forward(self,x):
        '''
        x : FloatTensor of shape (batch, time_frames, 84)

        returns logits : FloatTensor of shape (batch, time_frames, 84)
        '''

        x = x.unsqueeze(1)

        x = self.cnn(x)

        batch_size, channels, time_frames, freq_bins = x.size()

        x = x.permute(0,2,1,3).contiguous()

        x = x.view(batch_size, time_frames, channels*freq_bins)

        lstm_out, _ = self.bilstm(x)

        logits = self.fc(lstm_out)

        return logits



if __name__ == "__main__":
    model = PianoTranscriptArchitecture()
    dummy_input = torch.randn(8, 215, 84)
    output = model(dummy_input)
    print(f"Shape di input: {dummy_input.shape}")
    print(f"Shape di output: {output.shape}")

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