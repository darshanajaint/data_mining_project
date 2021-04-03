#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 11:48:56 2021

@author: ithier
"""

from torch.nn import BCEWithLogitsLoss
import torch.optim as optim
import argparse

from GRUModel import GRUModel
from data_util import *
from util import save_model, save_metrics

import time

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

        start = time.time()
        for batch in train_iterator:
            text = batch.text[1].to(device)
            labels = batch.label[1].to(device)

            output = model(text)
            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_epoch += loss.item()

        training_loss.append(train_loss_epoch)

        end = time.time()
        print("Epoch {:d} took {:.6f}s to train.".format(epoch, end - start))

        # want to evaluate on validation set after we've trained on all
        # training data that's available (i.e. all batches)
        start = time.time()
        val_loss_epoch = evaluate(val_iterator, model, criterion, device)
        validation_loss.append(val_loss_epoch)
        end = time.time()
        print("Epoch {:d} took {:.6f}s to validate.".format(epoch, end - start))

        # Keep track of epoch with minimum validation loss
        if val_loss_epoch < min_loss:
            min_loss = val_loss_epoch
            min_epoch = epoch

        # Save trained model
        save_model(model_path + "_epoch_{:d}.pt".format(epoch), model,
                   optimizer)
        save_metrics(metrics_path + "_epoch_{:d}.pt".format(epoch), epoch, 0, 0,
                     train_loss_epoch, val_loss_epoch)

        print("Finished epoch {:d}\n"
              "\tTotal training loss: {:.6f}\n"
              "\tTotal validation loss: {:.6f}"
              .format(epoch, train_loss_epoch, val_loss_epoch))

    print("Finished training!\n"
          "\tBest validation loss achieved after epoch: {:d}\n"
          "\tMin validation loss: {.6f}".format(min_epoch, min_loss))


def evaluate(data_loader, model, criterion, device):
    model.eval()

    loss = 0
    with torch.no_grad():
        for batch in data_loader:
            text = batch.text.to(device)
            labels = batch.label.to(device)

            output = model(text)
            loss += criterion(output, labels).item()

    return loss


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

    args = parser.parse_args()
    for arg in vars(args):
        print(arg, getattr(args, arg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    start = time.time()
    train_iter, val_iter, TEXT, _ = load_data(args.train_csv,
                                              args.max_vocab_size,
                                              args.batch_size, device)
    end = time.time()
    print("Time to load data: {:.6f}".format(end - start))

    model = GRUModel(input_size=args.input_size, hidden_size=args.hidden_size,
                     num_classes=2, text_field=TEXT, dropout=args.dropout,
                     bidirectional=args.bidirectional)

    train(model, train_iter, val_iter, args.epochs, device)


if __name__ == "__main__":
    main()
