# Metrics

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np

from dataset import MusicNetPianoDataset
from model import PianoTranscriptArchitecture
from utils import extract_cqt, get_device, load_checkpoint, load_config

from plots import (
    plot_precision_recall_threshold,
    plot_prob_distribution,
    plot_confusion_per_note,
    plot_piano_roll
)

from report import log_metrics

def evaluate():

    # load hyperparameters from the config gile
    config = load_config("configs/config.yaml")

    device = get_device()
    print(f"Metrics on :{device}")

    # -------- Dataset and Dataloader ----------
    # load test dataset
    test_dataset = MusicNetPianoDataset(split='test')
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # -------- Model ---------------------------
    # initialize and load the trained model
    model = PianoTranscriptArchitecture(
        input_features = config['model']['input_features'],
        hidden_size = config['model']['hidden_size'],
        lstm_layers = config['model']['lstm_layers'],
        dropout = config['model']['dropout']
    ).to(device)

    # load the weights computed after the train
    load_checkpoint(config["evaluation"]["checkpoint_path"], model, device=device)

    # model in evaluation mode
    model.eval()

    # lists for prediction, probabilities and labels
    all_probs = []
    all_labels = []
    track_info = [] # for save data for piano_roll

    # -------- Evalutaion loop -----------------
    # disable the gradient computation for speed
    with torch.no_grad():
        for batch in test_loader:
            waveforms = batch["waveform"]
            labels = batch["labels"].to(device)
            track_ids = batch["id"]

            # extract CQT like in the training
            cqt_list = [extract_cqt(wave.squeeze()).float() for wave in waveforms]
            inputs = torch.stack(cqt_list).to(device)

            # prediction of the model
            logits = model(inputs)

            # temporal alignment between pred and labels
            min_frames = min(logits.size(1), labels.size(1))
            logits = logits[:, :min_frames, :]
            labels = labels[:, :min_frames, :]

            # apply sigmoid function for obtain probability between 0 and 1
            probs = torch.sigmoid(logits)

            probs_np = probs.cpu().numpy()
            labels_np = labels.cpu().numpy()

            all_probs.append(probs_np.reshape(-1,84))
            all_labels.append(labels_np.reshape(-1,84))


            if len(track_info) < 5:
                for i in range(len(track_ids)):
                    track_info.append({
                        'id': track_ids[i],
                        'probs': probs[i].cpu().numpy(),
                        'labels': labels[i].cpu().numpy()
                    })

    # check
    if len(all_probs) == 0:
        print("Error: no data. Verify test dataset")
        return

    # concatenate the results
    all_probs = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)

    threshold = config['evaluation']['threshold']
    all_preds = (all_probs >= threshold).astype(np.float32)

    # --- Plot generation ---
    print("\nPlot Generation")

    # plot functions
    plot_precision_recall_threshold(all_probs,all_labels)
    plot_prob_distribution(all_probs, all_labels)
    plot_confusion_per_note(all_labels, all_probs, threshold=threshold)

    # piano roll
    for info in track_info:
        plot_piano_roll(
            info['labels'],
            info['probs'],
            track_id=info['id'],
            threshold=threshold
        )

    # metrics
    # average='micro' metrics calculated globally on all the frames
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='micro', zero_division=0
    )

    # accuracy : number of true predictions
    accuracy = (all_labels == all_preds).mean()

    # probs
    max_probs = all_probs.max()
    mean_probs = all_probs.mean()
    active_preds = all_preds.mean()
    active_labels = all_labels.mean()

    # saves into log
    log_metrics(accuracy, precision, recall, f1, max_probs, mean_probs, active_preds, active_labels, threshold)

    # prints
    print('\n--- Results of Frame-Level Evaluation ---')

    # Debug: print probability statistics to understand model output distribution
    print(f"Max prob  : {max_probs:.4f}")   
    print(f"Mean prob : {mean_probs:.4f}")
    print(f"% active preds : {active_preds:.4f}")
    print(f"% active labels: {active_labels:.4f}")


    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

if __name__ == "__main__":
    evaluate()