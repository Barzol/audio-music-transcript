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

class PianoTranscriptArchitecture(nn.Module):

    def __init__(
            self,
            #---------------------------------------------------
            input_features = 84,    # frequency bins Prima 84 -> cqt, 1025 STFT
            #---------------------------------------------------
            hidden_size = 128,      # size of LSTM ( output will be *2 because is Bidirectional)
            lstm_layers = 1,        
            dropout = 0.3           # dropout probability
            ):
        

        super(PianoTranscriptArchitecture, self).__init__()

        #----------------AGGIUNTO DA CLAUDE--------------------
        # ← NUOVO: comprime 1025 → 84 prima della CNN
        #self.input_proj = nn.Linear(input_features, 84)
        #------------------------------------------------------

        # ----- CNN MODULE -----
        # the CNN receives the CQT matrix and performs a frame by frame classification
        # expected input shape : (batch, channel=1, time_frames, bin_freq =84)

        self.cnn = nn.Sequential(
            # first convolutional block
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(32), # normalizes activations
            nn.ReLU(),
            # halves the dimension of the frequencies ( 84 -> 42 )
            nn.MaxPool2d(kernel_size=(1,2)),

            # second convolutional block
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(dropout), #dropout for prevent overfitting
            # halves the dimension of frequencies (42 -> 21)
            nn.MaxPool2d(kernel_size=(1,2)),            
        )

        # Calculates the outpu dimension of CNN for passing it to LSTM
        # 84 bin, after MaxPool they are 21, we multiply it for the 64 output channels
        #-----------------------------------------------------------
        cnn_out_freq = input_features // 4
        #cnn_out_freq = 84 // 4
        #-----------------------------------------------------------
        cnn_out_features = 64 * cnn_out_freq


        # ----- BIDIRECTIONAL LSTM MODULE -----
        # BiLSTM for the duration of notes in the frames
        # watches past and future for learn when a note starts and stops
        self.bilstm = nn.LSTM(
            input_size=cnn_out_features,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True
        )


        # ----- FINAL CLASSIFICATION ----- 
        # the problem is a multi-label classification so for each frame the model produces 84 probabilities
        # Fully connected layer for 84 probabilities
        # hidden_size * 2 because it's bidirectional
        self.fc = nn.Linear(hidden_size * 2, 84)


    def forward(self,x):
        '''
        x : FloatTensor of shape (batch, time_frames, 84)

        returns logits : FloatTensor of shape (batch, time_frames, 84)
        '''

        #----------------------------------------------------------------------------
        # x: (Batch, Frame, 1025)
        #x = self.input_proj(x)   # ← NUOVO: (Batch, Frame, 1025) → (Batch, Frame, 84)

        #x = torch.relu(x)        # ← aggiungi questa riga RELU
        #----------------------------------------------------------------------------


        # x enters with shape : (Batch, Frame, 84)
        # we add the channel dimension
        # input shape : (Batch, 1, Frame, 84)
        x = x.unsqueeze(1)

        # CNN
        # output shape : (Batch, 64, Frame, 21)
        x = self.cnn(x)

        # tensor for LSTM
        # LSTM wants a 3d input : (Batch, Time, Feature)
        batch_size, channels, time_frames, freq_bins = x.size()

        # 
        x = x.permute(0,2,1,3).contiguous()

        # Flatten the channel and frequencies in only one dimension "feature"
        x = x.view(batch_size, time_frames, channels*freq_bins)

        # BiLSTM
        # output shape : (Batch, Frame, 256)
        lstm_out, _ = self.bilstm(x)

        # linear layer
        # outputs shape : (Batch, Frame, 84)
        logits = self.fc(lstm_out)

        return logits



# --- TEST DEL MODELLO ---
if __name__ == "__main__":
    # Test veloce per verificare le dimensioni dei tensori
    model = PianoTranscriptArchitecture()
    #---------------------------------------------------------------------------------------------------------
    dummy_input = torch.randn(8, 215, 84) # Batch da 8 campioni [cite: 36], 215 frame, 84 bin CQT [cite: 31]
    #dummy_input = torch.randn(8, 215, 1025) # Batch da 8 campioni [cite: 36], 215 frame, 1025 bin CQT [cite: 31]
    #---------------------------------------------------------------------------------------------------------
    output = model(dummy_input)
    print(f"Shape di input: {dummy_input.shape}")
    print(f"Shape di output: {output.shape}") # Dovrebbe essere [8, 215, 84]

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