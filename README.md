# audio-music-transcript
Machine Learning for Vision and Multimedia Group Project

# Objective
Design and implement a deep learning system capable of automatically transcribing monophonic (one note at a time) or polyphonic (multiple notes sounding
simultaneously) music recordings of a single musical instrument into symbolic
notation (e.g., MIDI). The model must learn to map audio waveforms or timefrequency representations (e.g., spectrogram, Mel, CQT) to sequences of pitch
(note frequency) and onset (note start and duration) events. Consider single
instrument recordings and focus on one instrument (e.g., piano, guitar, violin,
flute, etc.).

# Dataset
- Use a publicly available dataset of single-instrument performances that
provides aligned audio and symbolic annotations. Each sample should
include: audio recording (preferably isolated instrument stems); groundtruth pitch and onset annotations (e.g., MIDI note and event lists).

- Alternatively, create a synthetic dataset using software synthesizers such
as FluidSynth or Virtual Instrument plugins for Digital Audio Workstations. Render audio from MIDI files. Document the synthesis configuration to ensure reproducibility.

# Experimental Plan
- Phase 1: Train a baseline model (e.g., CNN) that predicts pitch of single
or multiple notes from spectrograms.

- Phase 2: Extend the baseline model to retrieve both pitch and note onset/offset times, possibly via a multi-output architecture or post-processing
step.

- Phase 3: Perform qualitative and quantitative analysis of the model behavior comparing different time-frequency representations (e.g., STFT, Mel,
CQT). Evaluate model robustness to recording/synthesis variations (e.g.,
reverb, detuning, noise). Optionally, assess the contribution of temporal
sequence models (RNN, LSTM) relative to pure convolutional baselines.

# Example dataset
MAESTRO: https://magenta.tensorflow.org/datasets/maestro MusicNet: ht
tps://zenodo.org/records/5120004 Slakh: http://www.slakh.com/ NSynth:
https://magenta.tensorflow.org/datasets/nsynth.
