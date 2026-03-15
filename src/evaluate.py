# Metrics

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np

from dataset import MusicNetPianoDataset
from model import PianoTranscriptArchitecture
from utils import extract_cqt, get_device, load_checkpoint

def evaluate():
    device = get_device()
    print(f"Metrics on :{device}")

    # load test dataset
    test_dataset = MusicNetPianoDataset(split="test")
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # initialize and load the trained model
    model = PianoTranscriptArchitecture().to(device)
    # load the weights computed after the train
    load_checkpoint("checkpoints/best_model.pt", model, device=device)

    # model in evaluation mode
    model.eval()

    # lists for prediction and labels
    all_preds = []
    all_labels = []

    # disable the gradient computation for speed
    with torch.no_grad():
        for batch in test_loader:
            waveforms = batch["waveform"]
            labels = batch["labels"].to(device)

            # extract CQT like in the training
            cqt_list = [extract_cqt(wave) for wave in waveforms]
            inputs = torch.stack(cqt_list).to(device)

            # prediction of the model
            logits = model(inputs)

            # temporal alignment between pred and labels
            min_frames = min(logits.size(1), labels.size(1))
            logits = logits[:, :min_frames, :]
            labels = labels[:, :min_frames, :]

            # apply sigmoid function for obtain probability between 0 and 1
            probs = torch.sigmoid(logits)

            # apply the threshold of 0.5
            preds = (probs >= 0.5).float()

            # move the tensors on the CPU and convert them in array
            # flats the dimensions (batch*frames, 84) for calculate the metrics
            all_preds.append(preds.cpu().numpy().reshape(-1,84))
            all_labels.append(labels.cpu().numpy().reshape(-1, 84))

    # concatenate the results
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    # metrics
    # average='micro' metrics calculated globally on all the frames
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='micro')

    # accuracy : number of true predictions
    accuracy = accuracy_score(all_labels, all_preds)

    # prints
    print('\n--- Results of Frame-Level Evaluation ---')
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

if __name__ == "__main__":
    evaluate()