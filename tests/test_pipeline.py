import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
from torch.utils.data import DataLoader

from src.dataset import MusicNetPianoDataset
from src.model import PianoTranscriptArchitecture

# -------------------------------
# CONFIG TEST
# -------------------------------
FEATURE_TYPE = "stft"   # cambia in "cqt" per testare CQT
BATCH_SIZE = 2


# -------------------------------
# 1. TEST DATASET
# -------------------------------
def test_dataset():
    print("\n--- TEST DATASET ---")

    dataset = MusicNetPianoDataset(
        split="train",
        feature_type=FEATURE_TYPE
    )

    sample = dataset[0]

    features = sample["features"]
    labels = sample["labels"]

    print("Features shape:", features.shape)
    print("Labels shape:", labels.shape)

    assert features.ndim == 2, "Features devono essere 2D (frames, freq)"
    assert labels.ndim == 2, "Labels devono essere 2D (frames, notes)"

    print("Dataset OK ✅")


# -------------------------------
# 2. TEST DATALOADER
# -------------------------------
def test_dataloader():
    print("\n--- TEST DATALOADER ---")

    dataset = MusicNetPianoDataset(
        split="train",
        feature_type=FEATURE_TYPE
    )

    loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    batch = next(iter(loader))

    features = batch["features"]
    labels = batch["labels"]

    print("Batch features:", features.shape)
    print("Batch labels:", labels.shape)

    assert features.ndim == 3, "Batch features devono essere 3D"
    assert labels.ndim == 3, "Batch labels devono essere 3D"

    print("Dataloader OK ✅")


# -------------------------------
# 3. TEST MODELLO
# -------------------------------
def test_model():
    print("\n--- TEST MODEL ---")

    if FEATURE_TYPE == "cqt":
        input_features = 84
    elif FEATURE_TYPE == "stft":
        input_features = 1025
    else:
        raise ValueError("Feature type non valido")

    model = PianoTranscriptArchitecture(input_features=input_features)

    dummy = torch.randn(BATCH_SIZE, 215, input_features)

    output = model(dummy)

    print("Input shape:", dummy.shape)
    print("Output shape:", output.shape)

    assert output.shape[-1] == 84, "Output deve avere 84 note"

    print("Model OK ✅")


# -------------------------------
# 4. TEST FORWARD COMPLETO
# -------------------------------
def test_full_pipeline():
    print("\n--- TEST FULL PIPELINE ---")

    dataset = MusicNetPianoDataset(
        split="train",
        feature_type=FEATURE_TYPE
    )

    loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    batch = next(iter(loader))

    features = batch["features"]
    labels = batch["labels"]

    if FEATURE_TYPE == "cqt":
        input_features = 84
    else:
        input_features = 1025

    model = PianoTranscriptArchitecture(input_features=input_features)

    outputs = model(features)

    print("Features:", features.shape)
    print("Outputs:", outputs.shape)
    print("Labels:", labels.shape)

    # allineamento temporale
    min_frames = min(outputs.size(1), labels.size(1))
    outputs = outputs[:, :min_frames, :]
    labels = labels[:, :min_frames, :]

    assert outputs.shape == labels.shape, "Mismatch tra output e label"

    print("Full pipeline OK ✅")


# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    test_dataset()
    test_dataloader()
    test_model()
    test_full_pipeline()

    print("\n🎉 ALL TEST PASSED 🎉")