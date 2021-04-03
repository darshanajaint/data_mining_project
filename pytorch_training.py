#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 11:48:56 2021

@author: ithier
"""

from torch.nn import BCEWithLogitsLoss
import torch.optim as optim
import argparse
import numpy as np

from GRUModel import GRUModel
from data_util import *
from util import save_model, save_metrics
from sklearn.metrics import accuracy_score

"""
Other packages to download
https://pypi.org/project/spacy-pytorch-transformers/
https://clay-atlas.com/us/blog/2020/05/12/python-en-package-spacy-error/
"""


def train(model, train_iterator, val_iterator, num_epochs, device,
          model_path="", metrics_path=""):
    criterion = BCEWithLogitsLoss()
    optimizer = optim.Adam(params=model.parameters())

    training_loss = []
    validation_loss = []

    training_accuracy = []
    validation_accuracy = []

    train_labels = get_labels(train_iterator, device)

    min_loss = float("inf")
    min_epoch = -1

    if model_path == "":
        model_path = "./gru_models/gru_model"
    if metrics_path == "":
        metrics_path = "./gru_metrics/gru_metrics"

    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        train_loss_epoch = 0

        train_predictions = []
        for batch in train_iterator:
            text = batch.text.to(device)
            label = batch.label.to(device)

            output = model(text)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_epoch += loss.item()
            train_predictions.append(output.detach().cpu().numpy())

        train_predictions = np.asarray(train_predictions)
        train_acc_epoch = accuracy_score(train_labels, train_predictions)

        training_loss.append(train_loss_epoch)
        training_accuracy.append(train_acc_epoch)

        val_loss_epoch, val_acc_epoch = evaluate(val_iterator, model,
                                                 criterion, device)
        validation_loss.append(val_loss_epoch)
        validation_accuracy.append(val_acc_epoch)

        # Keep track of epoch with minimum validation loss
        if val_loss_epoch < min_loss:
            min_loss = val_loss_epoch
            min_epoch = epoch

            # Save best model so far
            save_model(model_path + "_epoch_{:d}.pt".format(epoch), model,
                       optimizer)
            save_metrics(metrics_path + "_epoch_{:d}.pt".format(epoch), epoch,
                         train_acc_epoch, val_acc_epoch, train_loss_epoch,
                         val_loss_epoch)

        # Compute training and validation accuracies.

        print("Finished epoch {:d}\n"
              "\tTraining accuracy: {:.6f}"
              "\tValidation accuracy: {:.6f}"
              "\tTotal training loss: {:.6f}\n"
              "\tTotal validation loss: {:.6f}"
              .format(epoch, train_acc_epoch, val_acc_epoch, train_loss_epoch,
                      val_loss_epoch))

    print("Finished training!\n"
          "\tBest validation loss achieved after epoch: {:d}\n"
          "\tMin validation loss: {:.6f}".format(min_epoch, min_loss))


def evaluate(data_loader, model, criterion, device):
    model.eval()

    true_labels = get_labels(data_loader, device)

    predictions = []
    loss = 0
    with torch.no_grad():
        for batch in data_loader:
            text = batch.text.to(device)
            labels = batch.label.to(device)

            output = model(text)
            loss += criterion(output, labels).item()

            predictions.append(output.cpu().numpy())

    predictions = np.asarray(predictions)
    accuracy = accuracy_score(true_labels, predictions)

    return loss, accuracy


def get_labels(iterator, device):
    labels = []
    for batch in iterator:
        label = batch.label.to(device)
        labels.append(label.cpu().numpy())
    return np.asarray(labels)


def str2bool(arg):
    if isinstance(arg, bool):
        return argparse
    if arg == "True":
        return True
    elif arg == "False":
        return False
    else:
        raise argparse.ArgumentTypeError("boolean expected")


def main():
    parser = argparse.ArgumentParser(description='Train neural network')
    parser.add_argument('--batch_size', type=int,
                        help='batch size to be used in training and evaluation')
    parser.add_argument('--train_csv', type=str,
                        help='csv filename and path for the train dataset')
    parser.add_argument('--lr', type=float,
                        help='learning rate for training')
    parser.add_argument('--epochs', type=int,
                        help='number of epochs to train for')
    parser.add_argument('--max_vocab_size', type=int, default=25000,
                        help='max vocabulary size for embedding')
    parser.add_argument('--input_size', type=int,
                        help='dimension of input layer to GRU')
    parser.add_argument('--hidden_size', type=int,
                        help='dimension of number of hidden units in GRU')
    parser.add_argument('--dropout', type=float,
                        help='decimal number of percentage of dropout to apply '
                             'in model')
    parser.add_argument('--bidirectional', type=str2bool,
                        help='True if GRU should be bidirectional, False '
                             'otherwise')
    parser.add_argument('--build_vocab', type=bool,
                        help='Whether to build vocab from train set or to use '
                             'prebuilt vocab. If False, ensure the '
                             'files with the text and label vocab exist and '
                             'are specified under their respective tags. '
                             'Default True.',
                        default=True)
    parser.add_argument('--text_vocab_path', type=str,
                        help='Path where text vocab will be saved (if built '
                             'from data set) or loaded from. Default ' 
                             './text_vocab.pickle.',
                        default='./text_vocab.pickle')
    parser.add_argument('--label_vocab_path', type=str,
                        help='Path where label vocab will be saved (if built '
                             'from data set) or loaded from. Default '
                             './label_vocab.pickle.',
                        default='./label_vocab.pickle')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_iter, val_iter, TEXT, _ = load_data(
        args.train_csv,
        args.max_vocab_size,
        args.batch_size,
        device,
        build_vocab=args.build_vocab,
        text_vocab_path=args.text_vocab_path,
        label_vocab_path=args.label_vocab_path
    )

    model = GRUModel(input_size=args.input_size, hidden_size=args.hidden_size,
                     text_field=TEXT, dropout=args.dropout,
                     bidirectional=args.bidirectional)

    model.to(device)

    train(model, train_iter, val_iter, args.epochs, device)


if __name__ == "__main__":
    main()
