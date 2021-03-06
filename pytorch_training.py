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
from util import str2bool
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
import torch
import torch.backends.cudnn

"""
Other packages to download
https://pypi.org/project/spacy-pytorch-transformers/
https://clay-atlas.com/us/blog/2020/05/12/python-en-package-spacy-error/
"""


def main():
    parser = argparse.ArgumentParser(description='Train neural network')
    parser.add_argument('--batch_size', type=int,
                        help='batch size to be used in training and evaluation', 
                        default=128)
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
    parser.add_argument('--test_after_train', type=str2bool,
                        help='Whether to run the test set immediately after '
                             'training.', default=True)
    parser.add_argument('--test', type=str2bool,
                        help='Whether to open the model in test mode or train '
                             'mode. Default False.', default=False)
    parser.add_argument('--test_csv', type=str,
                        help='Path to test data. Default empty. Must be set '
                             'if test or test_after_train is True.', default='')
    parser.add_argument('--model_load_path', type=str,
                        help='Path from which to load a pretrained model. '
                             'Must beset if test is True.', default='')
    parser.add_argument('--test_metrics_save_path', type=str,
                        help='Path where the test metrics will be saved. Must '
                             'be set if test or test_after_train is True. '
                             'Default ./test_metrics.pt',
                        default='./test_metrics.pt')

    args = parser.parse_args()

    if args.test and (args.test_csv == '' or args.model_load_path == '' or
                      args.test_metrics_save_path == ''):
        raise ValueError('Cannot have empty test_csv or model_load_path '
                         'or test_metrics_save_path when testing.')
    elif args.test_after_train and (args.test_csv == '' or
                                    args.test_metrics_save_path == ''):
        raise ValueError('Cannot have empty test_csv or '
                         'test_metrics_save_path when testing after training.')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    SEED = 0
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    training_data = read_csv(args.train_csv, train_val_split=args.validation)

    fields, TEXT, _ = setup_fields(training_data, args.max_vocab_size)
    
    model = GRUModel(input_size=args.input_size, hidden_size=args.hidden_size,
                     text_field=TEXT, dropout=args.dropout,
                     bidirectional=args.bidirectional)
    model.to(device)

    if not args.test:
        optimizer = Adam(params=model.parameters(), lr=args.lr)
        criterion = BCEWithLogitsLoss()
        criterion.to(device)
        
        model = ModelUtil(model, args.batch_size, fields, device, optimizer,
                      criterion, args.model_save_path, args.metrics_save_path)
    
    
        model.fit(training_data, args.epochs, args.validation,
                  args.save_final_model)
        
        if args.test_after_train:
            model.load_model(args.model_save_path)
    else:
        model = ModelUtil(model, 128, fields, device, None,
                      None, None, None)
        
        model.load_model(args.model_load_path)

    if args.test_after_train or args.test:
        print(args.test_metrics_save_path)
        test_data = read_csv(args.test_csv)
        # class_predictions = model.predict_class(test_data, False)
        # class_probabilities = model.predict_prob(test_data, False)
        test_accuracy, test_labels, test_predictions, test_probabilities = \
            model.accuracy_score(test_data, False)

        state = {
            'labels': test_labels,
            'predictions':  test_predictions,
            'probabilities': test_probabilities,
            'accuracy': test_accuracy
        }
        model.save_metrics(state, args.test_metrics_save_path)


if __name__ == "__main__":
    main()
