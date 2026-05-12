# generates all report visualizations
#
# 1. plot_loss_curve
# 2. plot_precision_recall
# 3. plot_piano_roll
# 4. plot_confusion_per note
# 5. plot_prob_distribution

import os
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches

from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support
from pathlib import Path

from dataset import MusicNetPianoDataset
from model import PianoTranscriptArchitecture
from utils import extract_cqt, get_device, load_checkpoint, load_config


# ouput directory
PLOTS_DIR = Path(__file__).parent.parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# midi note names for axis
MIDI_MIN = 33               # A1
MIDI_MAX = MIDI_MIN + 84    # C8


def midi_to_name(midi_number):
    """Converts a MIDI note number to a human-readable name e.g. 60 → C4."""
    names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (midi_number // 12) - 1
    name   = names[midi_number % 12]
    return f"{name}{octave}"


# -------- LOSS CURVE --------


def plot_loss_curve(train_losses, save=True):

    '''
    Plots the training loss over epochs.
 
    Args:
        train_losses (list of float): average loss value per epoch
        save         (bool)         : if True, saves the plot to plots/loss_curve.png 
    '''

    fig, ax = plt.subplots(figsize=(10,5))

    epochs = range(1, len(train_losses) + 1)
    
    ax.plot(
        epochs, 
        train_losses, 
        color='steelblue',
        linewidth = 2,
        label = 'Train Loss'
    )
    
    best_epoch = int(np.argmin(train_losses)) + 1
    best_loss = min(train_losses)
    
    ax.axvline(
        x=best_epoch,
        color='red',
        linestyle='--',
        alpha=0.6,
        label=f'Best epoch ({best_epoch})'
    )

    ax.scatter( 
        [best_epoch], 
        [best_loss], 
        color='red', 
        zorder = 5 
    )

    ax.set_title('Training Loss Curve', fontsize = 14, fontweight='bold')

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")

    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save:
        path = PLOTS_DIR / 'loss_curve.png'
        plt.savefig(path, dpi=150)
    
    # plt.show()
    plt.close()


# -------- PRECISION / RECALL / F1 vs THRESHOLD --------


def plot_precision_recall_threshold(all_probs, all_labels, save=True):
    '''
    Plots Precision, Recall, and F1-Score as a function of the sigmoid threshold.
    Useful for choosing the optimal threshold for note activation.
 
    Args:
        all_probs  (np.ndarray): sigmoid probabilities, shape (N, 84)
        all_labels (np.ndarray): ground truth binary labels, shape (N, 84)
        save       (bool)      : if True, saves to plots/precision_recall_threshold.png
    '''

    thresholds = np.arange(0.05, 0.95, 0.05)
    precision, recalls, f1s = [], [], []

    for t in thresholds:
        preds = (all_probs >= t).astype(np.float32)

        p,r,f,_ = precision_recall_fscore_support(
            all_labels,
            preds,
            average='micro',
            zero_division=0
        )
        precision.append(p)
        recalls.append(r)
        f1s.append(f)


    best_idx = int(np.argmax(f1s))
    best_threshold = thresholds[best_idx]
    best_f1 = f1s[best_idx]

    fig, ax = plt.subplots(figsize=(10,5))
    
    ax.plot(
        thresholds,
        precision,  
        color='royalblue',
        linewidth = 2,
        label = 'Precision'
    )

    ax.plot(
        thresholds,
        recalls, 
        color='darkorange',
        linewidth = 2,
        label = 'Recalls'
    )

    ax.plot(
        thresholds,
        f1s,  
        color='green',
        linewidth = 2,
        label = 'F1-Score'
    )

    ax.axvline(
        x=best_threshold,
        color='red',
        linestyle='--',
        alpha=0.6,
        label=f'Best threshold ({best_threshold:.2f})'
    )

    ax.set_title(
        'Precision, Recall, F1-Score vs Threshold', 
        fontsize = 14, 
        fontweight='bold'
    )

    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")

    ax.set_xlim(0,1)
    ax.set_ylim(0,1)

    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save:
        path = PLOTS_DIR / 'precision_recall_threshold.png'
        plt.savefig(path, dpi=150)
    
    # plt.show()
    plt.close()


# -------- PIANO ROLL --------


def plot_piano_roll(labels, preds, track_id="sample", threshold=0.3, save=True):


    binary_preds = (preds >= threshold).astype(np.float32)

    time_frames = labels.shape[0]
    rgb = np.zeros((84, time_frames,3), dtype=np.float32)

    gt = labels.T.astype(bool)
    pred = binary_preds.T.astype(bool)

    tp = gt & pred
    rgb[tp, 0] = 0.8
    rgb[tp, 2] = 0.8

    fn = gt & ~pred
    rgb[fn, 2] = 0.9

    fig, ax = plt.subplots(figsize=(10,5))

    ax.imshow(
        rgb,
        aspect='auto',
        origin='lower',
        extent=[0, time_frames, MIDI_MIN, MIDI_MAX]
    )

    c_notes = [m for m in range(MIDI_MIN, MIDI_MAX + 1) if (m % 12) == 0]

    ax.set_yticks([m - MIDI_MIN for m in c_notes])
    
    # Sostituisci o aggiungi questo dopo imshow:
    ax.set_ylim(MIDI_MIN, MIDI_MAX)

    ax.set_yticklabels([midi_to_name(m) for m in c_notes])

    ax.set_title(
        f'Piano Roll - Track {track_id}', 
        fontsize = 14, 
        fontweight='bold'
    )

    ax.set_xlabel("CQT Frame")
    ax.set_ylabel("Note")

    legend_handles = [
        mpatches.Patch(color=(0, 0, 0.9), label = 'Ground Truth (missed)'),
        mpatches.Patch(color=(0.9, 0, 0), label = 'Predicted (false alarm)'),
        mpatches.Patch(color=(0.8, 0, 0.8), label = 'True Positive'),
    ]
    ax.legend(
        handles=legend_handles,
        loc='upper right',
        fontsize=8
    )

    plt.tight_layout()

    if save:
        path = PLOTS_DIR / f'piano_roll_{track_id}.png'
        plt.savefig(path, dpi=150)
    
    # plt.show()
    plt.close()



# -------- PER-Note CONFUSION --------


def plot_confusion_per_note(all_labels, all_preds, threshold=0.3, save=True):

    '''
    For each of the 84 piano notes, shows the count of:
        - True Positives  (correctly detected notes)
        - False Positives (notes predicted but not active)
        - False Negatives (active notes missed by the model)
 
    Args:
        all_labels (np.ndarray): ground truth binary labels, shape (N, 84)
        all_preds  (np.ndarray): predicted probabilities,    shape (N, 84)
        threshold  (float)     : activation threshold
        save       (bool)      : if True, saves to plots/confusion_per_note.png    
    '''
    binary_preds = (all_preds >= threshold).astype(np.float32)

    tp = ((binary_preds == 1) & (all_labels == 1)).sum(axis=0)
    fp = ((binary_preds == 1) & (all_labels == 0)).sum(axis=0)
    fn = ((binary_preds == 0) & (all_labels == 1)).sum(axis=0)

    note_indices = np.arange(84)
    midi_numbers = note_indices + MIDI_MIN

    tick_positions = note_indices[::12]
    tick_labels = [midi_to_name(midi_numbers[i]) for i in tick_positions]

    fig, ax = plt.subplots(figsize=(10,5))

    ax.bar(
        note_indices, 
        tp, 
        color='green', 
        alpha=0.8, 
        label='True Positive'
    )

    ax.bar(
        note_indices, 
        fp, 
        bottom=tp, 
        color='red', 
        alpha=0.5, 
        label='False Positive'
    )

    ax.bar(
        note_indices, 
        fn, 
        bottom=tp+fp, 
        color='steelblue', 
        alpha=0.5, 
        label='False Negative'
    )


    ax.set_title(
        'Confusion matrix per note',
        fontsize = 14,
        fontweight = 'bold'
    )

    ax.set_xlabel('Note Index (A1 to C8)')
    ax.set_ylabel('Frame Count')
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.legend()

    plt.tight_layout()

    if save:
        path = PLOTS_DIR / 'confusion_per_note.png'
        plt.savefig(path, dpi=150)
    
    # plt.show()
    plt.close()


# -------- PROBABILITY DISTRIBUTION --------


def plot_prob_distribution(all_probs, all_labels, save=True):
    '''
    Plots the histogram of sigmoid probabilities for active vs inactive notes.
    Ideally the two distributions should be well separated.
    If they overlap heavily, the model struggles to distinguish notes.
 
    Args:
        all_probs  (np.ndarray): sigmoid probabilities,      shape (N, 84)
        all_labels (np.ndarray): ground truth binary labels, shape (N, 84)
        save       (bool)      : if True, saves to plots/prob_distribution.png
    '''

    probs_flat = all_probs.flatten()
    labels_flat = all_labels.flatten().astype(bool)

    # 'note on' probabilites
    active_probs = probs_flat[labels_flat]
    # 'note off' probabilities
    inactive_probs = probs_flat[~labels_flat]

    fig, ax = plt.subplots(figsize=(10,5))

    ax.hist(
        inactive_probs, 
        bins=50, 
        color='steelblue',
        label = 'Inactive notes', 
        density = True,
        alpha=0.5
    )

    if len(active_probs) > 0:
        ax.hist(
            active_probs, 
            bins=50, 
            color='darkorange', 
            label='Active notes',
            density=True, 
            alpha=0.5
        )

    ax.set_title(
        'Probability distribution : Active vs Inactive',
        fontsize = 14,
        fontweight = 'bold'
    )

    ax.set_xlabel('Sigmoid Probability')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save:
        path = PLOTS_DIR / 'prob_distribution.png'
        plt.savefig(path, dpi=150)
    
    # plt.show()
    plt.close()

