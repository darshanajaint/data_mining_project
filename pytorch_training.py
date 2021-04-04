#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 11:48:56 2021

@author: ithier
"""

import argparse
from GRUModel import GRUModel
from ModelUtil import ModelUtil
from data_util import *
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss

"""
Other packages to download
https://pypi.org/project/spacy-pytorch-transformers/
https://clay-atlas.com/us/blog/2020/05/12/python-en-package-spacy-error/
"""


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
    parser.add_argument('--bidirectional', type=bool,
                        help='True if GRU should be bidirectional, False '
                             'otherwise')
    parser.add_argument('--model_save_path', type=str,
                        help='Path to save model. Will have _epoch_epoch#.pt '
                             'appended to it. Default ./gru_models',
                        default='./gru_models')
    parser.add_argument('--metrics_save_path', type=str,
                        help='Path to save training metrics. Default '
                             './gru_metrics.pt',
                        default='./gru_metrics.pt')
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

    train_iter, val_iter, (fields, TEXT, LABEL) = load_train_val_data(
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
    optimizer = Adam(params=model.parameters(), lr=args.lr)
    criterion = BCEWithLogitsLoss()

    model = ModelUtil(model, args.batch_size, fields, device, optimizer,
                      criterion, args.model_save_path, args.metrics_save_path)

    model.fit(train_iter, args.epochs, val_iter)


if __name__ == "__main__":
    main()
