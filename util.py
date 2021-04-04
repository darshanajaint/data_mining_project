import torch
import numpy as np


def get_labels(iterator, device):
    labels = []
    for batch in iterator:
        label = batch.label.to(device)
        labels += list(label.cpu().numpy())
    return np.asarray(labels)


def load_metrics(path):
    return torch.load(path)
