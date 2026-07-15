
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


def evaluate():

    config = load_config("configs/config.yaml")

    device = get_device()
    print(f"Evaluation on: {device}")

    test_dataset = MusicNetPianoDataset(
        csv_file=config['dataset']['csv_file'],
        data_dir=config['dataset']['data_dir'],
        split='test',
    )
    print(f"Tracce di test trovate: {len(test_dataset)}")
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    model = PianoTranscriptArchitecture(
        input_features=config['model']['input_features'],
        dropout=config['model']['dropout'],
    ).to(device)

    load_checkpoint("checkpoints/best_model.pt", model, device=device)
    model.eval()

    all_probs  = []
    all_labels = []
    track_info = []

    with torch.no_grad():
        for batch in test_loader:
            waveforms = batch["waveform"]
            labels    = batch["labels"].to(device)
            track_ids = batch["id"]

            cqt_list = [extract_cqt(wave) for wave in waveforms]
            inputs   = torch.stack(cqt_list).to(device)

            logits = model(inputs)

            min_frames = min(logits.size(1), labels.size(1))
            logits = logits[:, :min_frames, :]
            labels = labels[:, :min_frames, :]

            probs = torch.sigmoid(logits)

            probs_np  = probs.cpu().numpy()
            labels_np = labels.cpu().numpy()

            all_probs.append(probs_np.reshape(-1, 84))
            all_labels.append(labels_np.reshape(-1, 84))

            for i in range(len(track_ids)):
                track_info.append({
                    'id':     track_ids[i],
                    'probs':  probs[i].cpu().numpy(),
                    'labels': labels[i].cpu().numpy(),
                })

    if len(all_probs) == 0:
        print("Errore: nessun dato nel test set. Controlla il dataset.")
        return

    all_probs  = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)

    threshold = config['evaluation']['threshold']
    all_preds = (all_probs >= threshold).astype(np.float32)

    print("\nGenerazione plot...")
    plot_precision_recall_threshold(all_probs, all_labels)
    plot_prob_distribution(all_probs, all_labels)
    plot_confusion_per_note(all_labels, all_probs, threshold=threshold)

    for info in track_info:
        plot_piano_roll(
            info['labels'],
            info['probs'],
            track_id=info['id'],
            threshold=threshold,
        )

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='micro', zero_division=0
    )

    accuracy      = (all_labels == all_preds).mean()
    max_probs     = all_probs.max()
    mean_probs    = all_probs.mean()
    active_preds  = all_preds.mean()
    active_labels = all_labels.mean()

    log_metrics(
        accuracy, precision, recall, f1,
        max_probs, mean_probs, active_preds, active_labels, threshold,
    )

    print('\n── Risultati – Frame-Level Evaluation ───────────────────')
    print(f"  Max prob       : {max_probs:.4f}")
    print(f"  Mean prob      : {mean_probs:.4f}")
    print(f"  % active preds : {active_preds:.4f}")
    print(f"  % active labels: {active_labels:.4f}")
    print()
    print(f"  Accuracy  : {accuracy:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print('─────────────────────────────────────────────────────────')


if __name__ == "__main__":
    evaluate()