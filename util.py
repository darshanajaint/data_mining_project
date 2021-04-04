import torch
import numpy as np
import argparse


def get_labels(iterator, device):
    labels = []
    for batch in iterator:
        label = batch.label.to(device)
        labels += list(label.cpu().numpy())
    return np.asarray(labels)


def load_metrics(path):
    return torch.load(path)


def str2bool(arg):
    if isinstance(arg, bool):
        return argparse
    if arg == "True":
        return True
    elif arg == "False":
        return False
    else:
        raise argparse.ArgumentTypeError("boolean expected")

