
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support
import numpy as np

from dataset import MaestroDataset
from model   import PianoTranscriptArchitecture
from utils   import extract_cqt, get_device, load_checkpoint, load_config, generate_test_samples

from plots import (
    plot_precision_recall_threshold,
    plot_prob_distribution,
    plot_confusion_per_note,
    plot_piano_roll
)
from report import log_metrics


def evaluate():

    config = load_config()
    device = get_device()
    print(f"Evaluation on: {device}")

    test_dataset = MaestroDataset(split='test')
    test_loader  = DataLoader(test_dataset,
                              batch_size=config['training']['batch_size'],
                              shuffle=False, num_workers=0)

    model = PianoTranscriptArchitecture(
        input_features=config['model']['input_features'],
        dropout=config['model']['dropout']
    ).to(device)

    load_checkpoint(config['evaluation']['checkpoint_path'], model, device=device)
    model.eval()

    all_pitch_probs  = []
    all_onset_probs  = []
    all_offset_probs = []
    all_pitch_labels  = []
    all_onset_labels  = []
    all_offset_labels = []
    track_info = []

    with torch.no_grad():
        for batch in test_loader:
            waveforms      = batch['waveform']
            labels         = batch['labels'].to(device)
            onset_labels   = batch['onset_labels'].to(device)
            offset_labels  = batch['offset_labels'].to(device)
            track_ids      = batch['id']

            cqt_list = [
                extract_cqt(wave.squeeze(),
                            hop_length=config['dataset']['hop_length']).float()
                for wave in waveforms
            ]
            inputs = torch.stack(cqt_list).to(device)

            out_pitch, out_onset, out_offset = model(inputs)

            T = min(out_pitch.size(1), labels.size(1))
            out_pitch  = out_pitch[:, :T, :]
            out_onset  = out_onset[:, :T, :]
            out_offset = out_offset[:, :T, :]
            labels        = labels[:, :T, :]
            onset_labels  = onset_labels[:, :T, :]
            offset_labels = offset_labels[:, :T, :]

            p_probs  = torch.sigmoid(out_pitch).cpu().numpy()
            on_probs = torch.sigmoid(out_onset).cpu().numpy()
            off_probs= torch.sigmoid(out_offset).cpu().numpy()

            all_pitch_probs.append(p_probs.reshape(-1, 84))
            all_onset_probs.append(on_probs.reshape(-1, 84))
            all_offset_probs.append(off_probs.reshape(-1, 84))
            all_pitch_labels.append(labels.cpu().numpy().reshape(-1, 84))
            all_onset_labels.append(onset_labels.cpu().numpy().reshape(-1, 84))
            all_offset_labels.append(offset_labels.cpu().numpy().reshape(-1, 84))

            if len(track_info) < 3:
                for i in range(len(track_ids)):
                    if len(track_info) < 3:
                        track_info.append({
                            'id':     track_ids[i],
                            'probs':  p_probs[i],
                            'labels': labels[i].cpu().numpy()
                        })

    if not all_pitch_probs:
        print("Error: empty test set.")
        return

    all_pitch_probs   = np.vstack(all_pitch_probs)
    all_onset_probs   = np.vstack(all_onset_probs)
    all_offset_probs  = np.vstack(all_offset_probs)
    all_pitch_labels  = np.vstack(all_pitch_labels)
    all_onset_labels  = np.vstack(all_onset_labels)
    all_offset_labels = np.vstack(all_offset_labels)

    threshold        = config['evaluation']['threshold']
    thresh_onset     = config['evaluation'].get('threshold_onset',  threshold)
    thresh_offset    = config['evaluation'].get('threshold_offset', threshold)

    pitch_preds  = (all_pitch_probs  >= threshold).astype(np.float32)
    onset_preds  = (all_onset_probs  >= thresh_onset).astype(np.float32)
    offset_preds = (all_offset_probs >= thresh_offset).astype(np.float32)

    print("\nGenerating plots...")
    plot_precision_recall_threshold(all_pitch_probs, all_pitch_labels)
    plot_prob_distribution(all_pitch_probs, all_pitch_labels)
    plot_confusion_per_note(all_pitch_labels, all_pitch_probs, threshold=threshold)
    generate_test_samples(track_info, plot_piano_roll, threshold)

    def compute_f1(labels, preds):
        p, r, f, _ = precision_recall_fscore_support(
            labels, preds, average='micro', zero_division=0)
        return p, r, f

    p_p,  r_p,  f_p  = compute_f1(all_pitch_labels,  pitch_preds)
    p_on, r_on, f_on = compute_f1(all_onset_labels,  onset_preds)
    p_of, r_of, f_of = compute_f1(all_offset_labels, offset_preds)

    accuracy     = (all_pitch_labels == pitch_preds).mean()
    max_prob     = all_pitch_probs.max()
    mean_prob    = all_pitch_probs.mean()
    active_preds = pitch_preds.mean()
    active_labels= all_pitch_labels.mean()

    log_metrics(accuracy, p_p, r_p, f_p,
                max_prob, mean_prob, active_preds, active_labels, threshold,
                f1_onset=f_on, f1_offset=f_of)

    print('\n--- Frame-Level Evaluation Results ---')
    print(f"  Pitch  — P: {p_p:.4f}  R: {r_p:.4f}  F1: {f_p:.4f}")
    print(f"  Onset  — P: {p_on:.4f}  R: {r_on:.4f}  F1: {f_on:.4f}")
    print(f"  Offset — P: {p_of:.4f}  R: {r_of:.4f}  F1: {f_of:.4f}")
    print(f"  Accuracy      : {accuracy:.4f}")
    print(f"  Max prob      : {max_prob:.4f}")
    print(f"  Mean prob     : {mean_prob:.4f}")
    print(f"  % active preds: {active_preds:.4f}")
    print(f"  % active labels:{active_labels:.4f}")


if __name__ == '__main__':
    evaluate()