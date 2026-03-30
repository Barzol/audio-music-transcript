# audio-music-transcript
Machine Learning for Vision and Multimedia Group Project

# Objective
Design and implement a neural network capable of automatically transcribing monophonic (one note at a time) or polyphonic (multiple notes sounding simultaneously) music recordings of a single musical instrument into symbolic notation (e.g., MIDI).

The model must learn to map audio waveforms or timefrequency representations (e.g., spectrogram, Mel, CQT) to sequences of pitch (note frequency) and onset (note start and duration) events.

Consider single instrument recordings and focus on one instrument (e.g., piano, guitar, violin,
flute, etc.).

# Example dataset
MAESTRO: https://magenta.tensorflow.org/datasets/maestro 
MusicNet: https://zenodo.org/records/5120004 
Slakh: http://www.slakh.com/ 
NSynth: https://magenta.tensorflow.org/datasets/nsynth


# Instructions for running the code

Run the command
pip install -r requirements.txt
For install all the required packets 


## 📂 Project Structure

```text
audio-music-transcript/
├── .gitignore              # Files ignored by git (data, checkpoints, etc.)
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies
├── main.py                 # Main execution script
├── configs/
│   └── config.yaml         # Training and model configuration
├── plots/                  # Folder for plots
├── data/                   # Dataset generated locally
│   ├── raw/                # Contains filtered dataset on case of a multi instrument dataset
│   └── solo_piano.csv      # Metadata for the filtered dataset
├── scripts/                # Data preparation scripts
│   ├── download_and_filter.py  # Download and filter dataset
│   ├── build_dataset.py        # Builds the dataset folders
└── src/                    # Core PyTorch source code
    ├── dataset.py          # Defines the Dataset Class
    ├── model.py            # Defines the model architecture
    ├── train.py            # Training class
    ├── evaluate.py         # Evaluation class
    ├── plots.py            # Plot generation funcions for training and evaluation results
    └── utils.py            # Some common used functions
```

## File Descriptions

### Setup & Execution
- main.py : Runs the project
- requirements.txt : List of all required libraries.

### Scripts
- download_and_filter.py : Downloads the full MusicNet dataset via Kagglehub. It creates a solo_piano.csv file to filter only 'Solo Piano' tracks.
- build_dataset.py : Reads the CSV and copies only the required Solo Piano files into a local directory.

### Source Code
- utils.py : Contains common used functions.
- dataset.py : Dataset class.
- model.py : Defines the Architecture of the model, it is a CRNN with Bidirectional LSTM to model note duration over time.
- train.py : Training loop.
- evaluate.py : Evaluation metrics.
- plots.py : Code for plot generation.


## How to Run the Project

( Try also with 
```
py -m #etc
```
or simply
```
py #etc
```
)

1. Install Dependencies
```
pip install -r requirements.txt
```

2. Download and Filter Data
```
python main.py --download
```

3. Build Local Dataset
```
python main.py --build
```

4. Train the Model
```
python main.py --train
```

5. Evaluation and Plot Results
```
python main.py --evaluate
```
