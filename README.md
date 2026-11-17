# Automatic Piano Transcription with CNN + BiLSTM

A deep learning system for **Automatic Music Transcription (AMT)** of solo piano recordings. It converts audio into a **piano roll** (the set of notes active in every time frame) and, in its final version, also detects **note onsets and offsets**. The model was developed in three stages, from a plain CNN to a **CNN + bidirectional LSTM** with three output heads. It was evaluated on **MusicNet** and **MAESTRO v3.0.0** with two time-frequency representations, **CQT** and **STFT**.

> Group project for the course *Machine Learning for Vision and Multimedia*, Politecnico di Torino.
> Team: Davide Marzolla, Calogero Falco Abramo, Francesco Giuseppe Accarrino.

**Best result: F1 = 0.607** (frame-level pitch, MAESTRO, CNN + BiLSTM, STFT features, multi-output).

<p align="center">
  <img src="plots/piano_roll_1759.png" width="85%" alt="Predicted vs ground-truth piano roll">
</p>

---

## Highlights

- **Three-stage design**: CNN baseline → multi-output CNN (pitch, onset, offset) → CNN + BiLSTM with bidirectional temporal context.
- **Two datasets compared**: MusicNet (156 solo-piano tracks, mixed audio quality) and MAESTRO (about 200 h of Disklavier recordings with very precise alignment). Comparing them isolates how much the amount and quality of data matter.
- **CQT vs STFT study**: the representation that wins depends on how much data is available (see [Key findings](#key-findings)).
- **Class imbalance**: only 2–5% of piano-roll cells are active, so training uses a cost-sensitive weighted BCE (`pos_weight`) and the decision threshold is tuned separately for each output head.
- **Data augmentation ablation**: pitch shift, detuning, gain, Gaussian noise, synthetic reverb and SpecAugment-style masking, tested one at a time and in combination.

## Method

**Problem formulation.** Frame-level multi-label classification: for each frame the model predicts which of the N notes are active (N = 84 for MusicNet, 88 for MAESTRO, i.e. the full piano range).

**Preprocessing.** Random 7 s chunks → mono → resampled to 22.05 kHz → time-frequency matrix (hop 512) → magnitude in dB. The labels are three aligned binary matrices: piano roll, onset roll and offset roll.

| Representation | Details |
|---|---|
| **CQT** | 12 bins per octave, so each bin is exactly one MIDI note and lines up with the labels without any interpolation |
| **STFT** | `n_fft` 1024, linear bins cropped to the piano range (about 193 bins between 55 Hz and 4.2 kHz) |

**Architecture (final stage).**

```
Input (B, T, F) ──► 2 × [Conv2D → BatchNorm → ReLU → MaxPool (frequency axis only)]
                ──► BiLSTM (256 units per direction) → 512-d context for each frame
                ──► 3 heads: Linear(512,128) → ReLU → Dropout → Linear(128, N)
                     ├── pitch
                     ├── onset
                     └── offset
Loss = 1.0·BCE_pitch + 0.3·BCE_onset + 0.3·BCE_offset   (weighted BCE with pos_weight)
```

**Training.** Adam (learning rate 1e-3), batch size 12, gradient clipping 1.0, ReduceLROnPlateau scheduler driven by the validation loss, best checkpoint selected on validation loss. **Metrics**: micro-averaged precision, recall and F1 over every cell of the piano roll.

## Results

**Best F1 for each stage (frame-level pitch)**

| Dataset | Architecture | Features | Precision | Recall | **F1** |
|---|---|---|---|---|---|
| MusicNet | CNN | CQT | 0.139 | 0.517 | 0.219 |
| MusicNet | CNN + BiLSTM | CQT + pitch-shift augmentation | 0.425 | 0.611 | **0.501** |
| MAESTRO | CNN | CQT | 0.445 | 0.657 | 0.530 |
| MAESTRO | CNN + BiLSTM | STFT-1024 | 0.565 | 0.654 | 0.606 |
| MAESTRO | CNN + BiLSTM, multi-output | STFT-1024 | 0.543 | 0.689 | **0.607** |

**Multi-output model on MAESTRO (STFT, thresholds tuned for each head)**

| Task | Precision | Recall | F1 |
|---|---|---|---|
| Pitch | 0.543 | 0.689 | 0.607 |
| Onset | 0.418 | 0.664 | 0.513 |
| Offset | 0.195 | 0.168 | 0.181 |

<p align="center">
  <img src="https://github.com/Barzol/audio-music-transcript/blob/maestro/phase3/plots/loss_curve.png?raw=true" width="70%" alt="Training and validation loss on MAESTRO">
</p>

## Key findings

- **Data matters more than architecture.** MAESTRO beats MusicNet at every stage. The size and the alignment quality of the dataset are the main bottleneck.
- **Temporal context is the biggest architectural gain.** Adding the BiLSTM improves F1 by about **+0.28 on MusicNet** and **+0.08 on MAESTRO** compared to the CNN baseline.
- **CQT is not always better than STFT.** On the small MusicNet subset CQT wins clearly (0.465 vs 0.394 F1). On MAESTRO the order reverses (STFT 0.606 vs CQT 0.588): with enough data the network learns the mapping from linear frequencies to musical pitch by itself.
- **On small datasets, less augmentation works better.** Pitch shift is the only technique that helps with CQT (+0.036 F1), and Gaussian noise is the best one with STFT (+0.079 F1). Every combination of techniques lowered performance.
- **Offsets are the hardest part.** The end of a piano note fades out gradually, so offset F1 stays around 0.18, while onsets reach 0.51.

## Repository branches

Each dataset and stage of the project has its own branch with its code, configuration and logs:

| | MusicNet | MAESTRO |
|---|---|---|
| Stage 1 – CNN | [`musicnet/phase1`](../../tree/musicnet/phase1) | [`maestro/phase1`](../../tree/maestro/phase1) |
| Stage 2 – multi-output CNN | [`musicnet/phase2`](../../tree/musicnet/phase2) | [`maestro/phase2`](../../tree/maestro/phase2) |
| Stage 3 – CNN + BiLSTM | [`musicnet/phase3`](../../tree/musicnet/phase3) | [`maestro/phase3`](../../tree/maestro/phase3) |

## Tech stack

**Python** · **PyTorch** · torchaudio · librosa · NumPy · pandas · TensorBoard · PyYAML

## Project structure

```text
├── main.py                 # CLI entry point
├── configs/config.yaml     # Dataset, features, model, training & evaluation settings
├── scripts/                # Dataset download, filtering and local build
├── src/
│   ├── dataset.py          # PyTorch Dataset: chunking, features, label alignment
│   ├── model.py            # CNN / CNN + BiLSTM models
│   ├── train.py            # Training loop, scheduler, checkpointing
│   ├── evaluate.py         # Metrics and threshold tuning
│   ├── plots.py            # Training/evaluation plots
│   ├── report.py           # Run logging
│   └── utils.py            # Feature extraction and helpers
├── logs/                   # Logs of each experiment
└── plots/                  # Generated figures
```

## Getting started

```bash
git checkout maestro/phase3      # or any other branch from the table above
pip install -r requirements.txt

python main.py --download   # 1. download the dataset
python main.py --build      # 2. build the local solo-piano subset
python main.py --train      # 3. train (settings in configs/config.yaml)
python main.py --evaluate   # 4. evaluate and generate plots
```
