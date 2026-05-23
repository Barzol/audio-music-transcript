# evaluate.py  –  Fase 2: Multi-Output CNN
#
# Differenze rispetto alla Fase 1:
#   - Il modello restituisce (logit_pitch, logit_onset, logit_offset)
#   - Metriche separate per pitch, onset e offset
#   - Le threshold sono configurabili separatamente nel config
#   - report.py logga le metriche di tutti e tre gli output

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support

from dataset import MusicNetPianoDataset
from model import PianoTranscriptArchitecture
from utils import extract_cqt, get_device, load_checkpoint, load_config

from plots import (
    plot_precision_recall_threshold,
    plot_prob_distribution,
    plot_confusion_per_note,
    plot_piano_roll,
)
from report import log_metrics


def compute_metrics(all_probs, all_labels, threshold, name=""):
    """Calcola e stampa precision, recall, F1 per un output."""
    preds = (all_probs >= threshold).astype(np.float32)

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, preds, average='micro', zero_division=0
    )
    accuracy      = (all_labels == preds).mean()
    max_prob      = all_probs.max()
    mean_prob     = all_probs.mean()
    active_preds  = preds.mean()
    active_labels = all_labels.mean()

    print(f"\n── {name} (threshold={threshold}) ───────────────────────────")
    print(f"  Max prob       : {max_prob:.4f}")
    print(f"  Mean prob      : {mean_prob:.4f}")
    print(f"  % active preds : {active_preds:.4f}")
    print(f"  % active labels: {active_labels:.4f}")
    print(f"  Accuracy       : {accuracy:.4f}")
    print(f"  Precision      : {precision:.4f}")
    print(f"  Recall         : {recall:.4f}")
    print(f"  F1-Score       : {f1:.4f}")

    return accuracy, precision, recall, f1, max_prob, mean_prob, active_preds, active_labels


def evaluate():

    config = load_config("configs/config.yaml")
    device = get_device()
    print(f"Evaluation on: {device}")

    # ── Dataset e DataLoader ─────────────────────────────────────────────────
    test_dataset = MusicNetPianoDataset(
        csv_file=config['dataset']['csv_file'],
        data_dir=config['dataset']['data_dir'],
        split='test',
    )
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # ── Modello ──────────────────────────────────────────────────────────────
    model = PianoTranscriptArchitecture(
        input_features=config['model']['input_features'],
        dropout=config['model']['dropout'],
    ).to(device)

    load_checkpoint("checkpoints/best_model.pt", model, device=device)
    model.eval()

    # ── Raccolta predizioni ───────────────────────────────────────────────────
    all_probs_pitch,  all_labels_pitch  = [], []
    all_probs_onset,  all_labels_onset  = [], []
    all_probs_offset, all_labels_offset = [], []
    track_info = []

    with torch.no_grad():
        for batch in test_loader:
            waveforms = batch["waveform"]
            labels    = batch["labels"].to(device)
            onsets    = batch["onsets"].to(device)
            offsets   = batch["offsets"].to(device)
            track_ids = batch["id"]

            cqt_list = [extract_cqt(wave) for wave in waveforms]
            inputs   = torch.stack(cqt_list).to(device)

            logit_pitch, logit_onset, logit_offset = model(inputs)

            T = min(logit_pitch.size(1), labels.size(1))
            logit_pitch  = logit_pitch[:, :T, :]
            logit_onset  = logit_onset[:, :T, :]
            logit_offset = logit_offset[:, :T, :]
            labels   = labels[:, :T, :]
            onsets   = onsets[:, :T, :]
            offsets  = offsets[:, :T, :]

            probs_pitch  = torch.sigmoid(logit_pitch).cpu().numpy()
            probs_onset  = torch.sigmoid(logit_onset).cpu().numpy()
            probs_offset = torch.sigmoid(logit_offset).cpu().numpy()

            all_probs_pitch.append(probs_pitch.reshape(-1, 84))
            all_probs_onset.append(probs_onset.reshape(-1, 84))
            all_probs_offset.append(probs_offset.reshape(-1, 84))

            all_labels_pitch.append(labels.cpu().numpy().reshape(-1, 84))
            all_labels_onset.append(onsets.cpu().numpy().reshape(-1, 84))
            all_labels_offset.append(offsets.cpu().numpy().reshape(-1, 84))

            for i in range(len(track_ids)):
                track_info.append({
                    'id':     track_ids[i],
                    'probs':  probs_pitch[i],
                    'labels': labels[i].cpu().numpy(),
                })

    if len(all_probs_pitch) == 0:
        print("Errore: nessun dato nel test set.")
        return

    all_probs_pitch  = np.vstack(all_probs_pitch)
    all_probs_onset  = np.vstack(all_probs_onset)
    all_probs_offset = np.vstack(all_probs_offset)
    all_labels_pitch  = np.vstack(all_labels_pitch)
    all_labels_onset  = np.vstack(all_labels_onset)
    all_labels_offset = np.vstack(all_labels_offset)

    thr_pitch  = config['evaluation']['threshold_pitch']
    thr_onset  = config['evaluation']['threshold_onset']
    thr_offset = config['evaluation']['threshold_offset']

    # ── Plot (basati sul pitch, come in Fase 1) ───────────────────────────────
    print("\nGenerazione plot...")
    plot_precision_recall_threshold(all_probs_pitch, all_labels_pitch)
    plot_prob_distribution(all_probs_pitch, all_labels_pitch)
    plot_confusion_per_note(all_labels_pitch, all_probs_pitch, threshold=thr_pitch)

    for info in track_info:
        plot_piano_roll(info['labels'], info['probs'],
                        track_id=info['id'], threshold=thr_pitch)

    # ── Metriche ──────────────────────────────────────────────────────────────
    print('\n══════════════ Risultati Fase 2 – Multi-Output CNN ══════════════')

    pitch_metrics  = compute_metrics(all_probs_pitch,  all_labels_pitch,  thr_pitch,  name="PITCH")
    onset_metrics  = compute_metrics(all_probs_onset,  all_labels_onset,  thr_onset,  name="ONSET")
    offset_metrics = compute_metrics(all_probs_offset, all_labels_offset, thr_offset, name="OFFSET")

    print('═════════════════════════════════════════════════════════════════')

    # Logga tutti e tre gli output
    log_metrics(pitch_metrics, onset_metrics, offset_metrics,
                thr_pitch, thr_onset, thr_offset)


if __name__ == "__main__":
    evaluate()