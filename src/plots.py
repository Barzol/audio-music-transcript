
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support


PLOTS_DIR = Path(__file__).parent.parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


def midi_to_name(midi_number):
    names  = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (midi_number // 12) - 1
    name   = names[midi_number % 12]
    return f"{name}{octave}"



def plot_loss_curve(train_losses, val_losses=None, save=True):
    fig, ax = plt.subplots(figsize=(10, 5))
    epochs  = range(1, len(train_losses) + 1)

    ax.plot(epochs, train_losses, color='steelblue',  linewidth=2, label='Training Loss')
    if val_losses is not None:
        ax.plot(epochs, val_losses, color='darkorange', linewidth=2, label='Validation Loss')

    best_epoch = int(np.argmin(val_losses if val_losses else train_losses)) + 1
    best_loss  = min(val_losses if val_losses else train_losses)
    ax.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.6,
               label=f'Best epoch ({best_epoch})')
    ax.scatter([best_epoch], [best_loss], color='red', zorder=5)

    ax.set_title('Training Loss Curve', fontsize=14, fontweight='bold')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        plt.savefig(PLOTS_DIR / 'loss_curve.png', dpi=150)
    plt.close()



def plot_precision_recall_threshold(all_probs, all_labels, save=True):
    thresholds = np.arange(0.05, 0.95, 0.05)
    precisions, recalls, f1s = [], [], []

    for t in thresholds:
        preds = (all_probs >= t).astype(np.float32)
        p, r, f, _ = precision_recall_fscore_support(
            all_labels, preds, average='micro', zero_division=0)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)

    best_idx       = int(np.argmax(f1s))
    best_threshold = thresholds[best_idx]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(thresholds, precisions, color='royalblue',  linewidth=2, label='Precision')
    ax.plot(thresholds, recalls,    color='darkorange', linewidth=2, label='Recall')
    ax.plot(thresholds, f1s,        color='green',      linewidth=2, label='F1-Score')
    ax.axvline(x=best_threshold, color='red', linestyle='--', alpha=0.6,
               label=f'Best threshold ({best_threshold:.2f})')

    ax.set_title('Precision, Recall, F1-Score vs Threshold', fontsize=14, fontweight='bold')
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        plt.savefig(PLOTS_DIR / 'precision_recall_threshold.png', dpi=150)
    plt.close()



def plot_piano_roll(labels, preds, track_id="sample", threshold=0.3,
                   midi_min=21, save=True):
    """
    midi_min : MIDI number del primo bin (21 per MAESTRO A0, 33 per MusicNet A1)
    labels/preds: shape (T, num_notes)
    """
    num_notes    = labels.shape[1]
    midi_max     = midi_min + num_notes
    binary_preds = (preds >= threshold).astype(np.float32)
    time_frames  = labels.shape[0]

    rgb  = np.zeros((num_notes, time_frames, 3), dtype=np.float32)
    gt   = labels.T.astype(bool)
    pred = binary_preds.T.astype(bool)

    tp = gt & pred
    rgb[tp, 0] = 0.8
    rgb[tp, 2] = 0.8

    fn = gt & ~pred
    rgb[fn, 2] = 0.9

    fp = ~gt & pred
    rgb[fp, 0] = 0.9

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(rgb, aspect='auto', origin='lower',
              extent=[0, time_frames, midi_min, midi_max])

    c_notes     = [m for m in range(midi_min, midi_max + 1) if (m % 12) == 0]
    c_notes_idx = [m - midi_min for m in c_notes]
    ax.set_yticks(c_notes_idx)
    ax.set_yticklabels([midi_to_name(m) for m in c_notes])
    ax.set_ylim(midi_min, midi_max)

    ax.set_title(f'Piano Roll — Track {track_id}', fontsize=14, fontweight='bold')
    ax.set_xlabel("CQT Frame")
    ax.set_ylabel("Note")

    legend_handles = [
        mpatches.Patch(color=(0, 0, 0.9),   label='Ground Truth (missed)'),
        mpatches.Patch(color=(0.9, 0, 0),   label='Predicted (false alarm)'),
        mpatches.Patch(color=(0.8, 0, 0.8), label='True Positive'),
    ]
    ax.legend(handles=legend_handles, loc='upper right', fontsize=8)
    plt.tight_layout()

    if save:
        plt.savefig(PLOTS_DIR / f'piano_roll_{track_id}.png', dpi=150)
    plt.close()



def plot_confusion_per_note(all_labels, all_preds, threshold=0.3,
                            midi_min=21, save=True):
    """
    midi_min  : MIDI number del primo bin (21 MAESTRO, 33 MusicNet)
    all_labels/all_preds: shape (N, num_notes)
    """
    num_notes    = all_labels.shape[1]
    binary_preds = (all_preds >= threshold).astype(np.float32)

    tp = ((binary_preds == 1) & (all_labels == 1)).sum(axis=0)
    fp = ((binary_preds == 1) & (all_labels == 0)).sum(axis=0)
    fn = ((binary_preds == 0) & (all_labels == 1)).sum(axis=0)

    note_indices = np.arange(num_notes)
    midi_numbers = note_indices + midi_min

    tick_step      = 12
    tick_positions = note_indices[::tick_step]
    tick_labels    = [midi_to_name(midi_numbers[i]) for i in tick_positions]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(note_indices, tp,          color='green',     alpha=0.8, label='True Positive')
    ax.bar(note_indices, fp, bottom=tp,         color='red',       alpha=0.5, label='False Positive')
    ax.bar(note_indices, fn, bottom=tp + fp,    color='steelblue', alpha=0.5, label='False Negative')

    ax.set_title('Confusion matrix per note', fontsize=14, fontweight='bold')
    ax.set_xlabel(f'Note (MIDI {midi_min}–{midi_min + num_notes - 1})')
    ax.set_ylabel('Frame Count')
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.legend()
    plt.tight_layout()

    if save:
        plt.savefig(PLOTS_DIR / 'confusion_per_note.png', dpi=150)
    plt.close()



def plot_prob_distribution(all_probs, all_labels, save=True):
    probs_flat  = all_probs.flatten()
    labels_flat = all_labels.flatten().astype(bool)

    active_probs   = probs_flat[labels_flat]
    inactive_probs = probs_flat[~labels_flat]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(inactive_probs, bins=50, color='steelblue',  label='Inactive notes',
            density=True, alpha=0.5)
    if len(active_probs) > 0:
        ax.hist(active_probs, bins=50, color='darkorange', label='Active notes',
                density=True, alpha=0.5)

    ax.set_title('Probability Distribution: Active vs Inactive', fontsize=14, fontweight='bold')
    ax.set_xlabel('Sigmoid Probability')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        plt.savefig(PLOTS_DIR / 'prob_distribution.png', dpi=150)
    plt.close()