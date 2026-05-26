# evaluate.py  -  Phase 3: CNN + BiLSTM + Multi-Output
#
# Rispetto alla Phase 2 / Phase 3 Exp 10-13:
#   - extract_features() al posto di extract_cqt()
#   - get_input_features() ricava input_features automaticamente
#   - multi_output flag: se False, valuta solo pitch

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support

from dataset import MusicNetPianoDataset
from model import PianoTranscriptArchitecture
from utils import (
    extract_features, get_input_features,
    get_device, load_checkpoint, load_config,
)

from plots import (
    plot_precision_recall_threshold,
    plot_prob_distribution,
    plot_confusion_per_note,
    plot_piano_roll,
)
from report import log_metrics


def compute_metrics(all_probs, all_labels, threshold, name=""):
    preds = (all_probs >= threshold).astype(np.float32)

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, preds, average='micro', zero_division=0
    )
    accuracy      = (all_labels == preds).mean()
    max_prob      = all_probs.max()
    mean_prob     = all_probs.mean()
    active_preds  = preds.mean()
    active_labels = all_labels.mean()

    print(f"\n-- {name} (threshold={threshold}) ---------------")
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

    config       = load_config("configs/config.yaml")
    device       = get_device()
    feat_config     = config['features']
    sr           = config['dataset']['sample_rate']
    multi_output = config['model'].get('multi_output', False)

    print(f"Evaluation on : {device}")
    print(f"Feature type  : {feat_config['type'].upper()}")
    print(f"Multi-output  : {multi_output}")

    # Dataset
    test_dataset = MusicNetPianoDataset(
        csv_file=config['dataset']['csv_file'],
        data_dir=config['dataset']['data_dir'],
        split='test',
    )
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # Modello
    input_features = get_input_features(feat_config, sr=sr)
    model = PianoTranscriptArchitecture(
        input_features=input_features,
        dropout=config['model']['dropout'],
        hidden_size=config['model']['hidden_size'],
        lstm_layers=config['model']['lstm_layers'],
    ).to(device)

    load_checkpoint("checkpoints/best_model.pt", model, device=device)
    model.eval()

    # Raccolta predizioni
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

            inputs = torch.stack(
                [extract_features(w, feat_config, sr=sr) for w in waveforms]
            ).to(device)

            logit_pitch, logit_onset, logit_offset = model(inputs)

            T = min(logit_pitch.size(1), labels.size(1))
            logit_pitch  = logit_pitch[:, :T, :]
            labels       = labels[:, :T, :]

            probs_pitch = torch.sigmoid(logit_pitch).cpu().numpy()
            all_probs_pitch.append(probs_pitch.reshape(-1, 84))
            all_labels_pitch.append(labels.cpu().numpy().reshape(-1, 84))

            if multi_output:
                logit_onset  = logit_onset[:, :T, :]
                logit_offset = logit_offset[:, :T, :]
                onsets       = onsets[:, :T, :]
                offsets      = offsets[:, :T, :]

                probs_onset  = torch.sigmoid(logit_onset).cpu().numpy()
                probs_offset = torch.sigmoid(logit_offset).cpu().numpy()
                all_probs_onset.append(probs_onset.reshape(-1, 84))
                all_probs_offset.append(probs_offset.reshape(-1, 84))
                all_labels_onset.append(onsets.cpu().numpy().reshape(-1, 84))
                all_labels_offset.append(offsets.cpu().numpy().reshape(-1, 84))

            for i in range(len(track_ids)):
                track_info.append({
                    'id':     track_ids[i],
                    'probs':  probs_pitch[i],
                    'labels': labels[i].cpu().numpy(),
                })

    if not all_probs_pitch:
        print("Errore: nessun dato nel test set.")
        return

    all_probs_pitch  = np.vstack(all_probs_pitch)
    all_labels_pitch = np.vstack(all_labels_pitch)

    thr_pitch  = config['evaluation']['threshold_pitch']
    thr_onset  = config['evaluation']['threshold_onset']
    thr_offset = config['evaluation']['threshold_offset']

    # Plot pitch
    print("\nGenerazione plot...")
    plot_precision_recall_threshold(all_probs_pitch, all_labels_pitch)
    plot_prob_distribution(all_probs_pitch, all_labels_pitch)
    plot_confusion_per_note(all_labels_pitch, all_probs_pitch, threshold=thr_pitch)

    for info in track_info:
        plot_piano_roll(info['labels'], info['probs'],
                        track_id=info['id'], threshold=thr_pitch)

    # Metriche
    print('\n====== Risultati Phase 3 – CNN + BiLSTM ======')
    pitch_metrics = compute_metrics(all_probs_pitch, all_labels_pitch, thr_pitch, "PITCH")

    if multi_output and all_probs_onset:
        all_probs_onset  = np.vstack(all_probs_onset)
        all_probs_offset = np.vstack(all_probs_offset)
        all_labels_onset  = np.vstack(all_labels_onset)
        all_labels_offset = np.vstack(all_labels_offset)
        onset_metrics  = compute_metrics(all_probs_onset,  all_labels_onset,  thr_onset,  "ONSET")
        offset_metrics = compute_metrics(all_probs_offset, all_labels_offset, thr_offset, "OFFSET")
    else:
        # Placeholder quando multi_output=False
        onset_metrics  = (0, 0, 0, 0, 0, 0, 0, 0)
        offset_metrics = (0, 0, 0, 0, 0, 0, 0, 0)

    print('================================================')

    log_metrics(pitch_metrics, onset_metrics, offset_metrics,
                thr_pitch, thr_onset, thr_offset)


if __name__ == "__main__":
    evaluate()