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
    parser.add_argument('--model_save_path', type=str,
                        help='Path to save model. Will have _epoch_epoch#.pt '
                             'appended to it. Default ./gru_models',
                        default='./gru_models')
    parser.add_argument('--metrics_save_path', type=str,
                        help='Path to save training metrics. Default '
                             './gru_metrics.pt',
                        default='./gru_metrics.pt')
    parser.add_argument('--validation', type=str2bool,
                        help='Whether to train with validation data or not. '
                             'Default True.', default=True)
    parser.add_argument('--save_final_model', type=str2bool,
                        help='Whether to save the final model during '
                             'training. Default False.', default=False)
    parser.add_argument('--test_csv', type=str,
                        help='Path to test data. Default empty.', default='')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    training_data = read_csv(args.train_csv, train_val_split=args.validation)

    fields, TEXT, _ = setup_fields(training_data, args.max_vocab_size)

    model = GRUModel(input_size=args.input_size, hidden_size=args.hidden_size,
                     text_field=TEXT, dropout=args.dropout,
                     bidirectional=args.bidirectional)
    model.to(device)
    optimizer = Adam(params=model.parameters(), lr=args.lr)
    criterion = BCEWithLogitsLoss()
    criterion.to(device)

    model = ModelUtil(model, args.batch_size, fields, device, optimizer,
                      criterion, args.model_save_path, args.metrics_save_path)

    model.fit(training_data, args.epochs, args.validation,
              args.save_final_model)

    if args.test_csv != '':
        test_data = read_csv(args.test_csv)
        class_predictions = model.predict_class(test_data)
        class_probabilities = model.predict_prob(test_data)
        test_accuracy = model.accuracy_score(test_data)

        print("Class predictions:\n", class_predictions[:10])
        print("Class probabilities:\n", class_probabilities[:10])
        print("Test accuracy:", test_accuracy)


if __name__ == "__main__":
    main()
