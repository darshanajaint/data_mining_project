#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 11:48:56 2021

@author: ithier
"""

import csv
import torch
from torchtext.legacy import data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import argparse

"""
Other packages to download
https://pypi.org/project/spacy-pytorch-transformers/
https://clay-atlas.com/us/blog/2020/05/12/python-en-package-spacy-error/
"""

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, text_field, dropout, bidirectional=True):
        super(GRUModel, self).__init__()
        self.embedding = nn.Embedding(len(text_field.vocab), input_size)
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=False, 
                          bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)
        self.bidirectional = bidirectional
        if bidirectional:
            self.fc = nn.Linear(hidden_size * 2, num_classes)
        else:
            self.fc = nn.Linear(hidden_size, num_classes)
    def forward(self, text, text_length):
        text_embedded = self.embedding(text)
        # text_embedded = [sentence length, batch size, embedding dim]
        output, hidden = self.rnn(text_embedded)
      
        if self.bidirectional:
          # TODO check indices
          output = torch.cat((hidden[-2, :, :], hidden[-1,:, :]), dim=1)
          output = self.fc(output)
        else:
          output = self.fc(output[-1, :, :])
       
        output = self.dropout(output)
        
        output = torch.squeeze(output, 1)
        output = torch.sigmoid(output)
        
        return output

class DataFrameDataset(data.Dataset):
    def __init__(self, df, fields):
        self.examples = []
        for i, row in df.iterrows():
            label = row["Party"]
            text = row["tidy_tweet"]
            self.examples.append(data.Example.fromlist([text, label], fields))
        super().__init__(self.examples, fields)
    
    def __getitem__(self, i):
        return self.examples[i]

def run_training_loop(args):
    df = pd.read_csv(args.train_csv)
    df = df[["tidy_tweet", "Party"]]
    
    df.loc[df["Party"] == "Independent", "Party"] = "Democrat"
    le = LabelEncoder()
    df["Party"] = le.fit_transform(df["Party"].values)
    
    train_df, valid_df = train_test_split(df, test_size=0.1)
    
    SEED = 0
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    TEXT = data.Field(tokenize="spacy", include_lengths=True)
    LABEL = data.LabelField(dtype=torch.float)
    fields = [("text", TEXT), ("label", LABEL)]
    
    train_ds = DataFrameDataset(train_df, fields)
    val_ds = DataFrameDataset(valid_df, fields)
    
    TEXT.build_vocab(train_ds, max_size = args.max_vocab_size, 
                     vectors = "glove.6B.200d", 
                     unk_init = torch.Tensor.zero_)
    LABEL.build_vocab(train_ds)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_iterator, val_iterator = data.BucketIterator.splits(# Datasets for iterator to draw data from
                              (train_ds, val_ds),

                              # Tuple of train and validation batch sizes.
                              batch_sizes=(args.batch_size, args.batch_size),

                              # Device to load batches on.
                              device=device,

                              sort_key=lambda x: len(x.text),

                              # Sort all examples in data using `sort_key`.
                              sort=False,

                              # Shuffle data on each epoch run.
                              shuffle=True,

                              # Use `sort_key` to sort examples in each batch.
                              sort_within_batch=True,
                              )
    
    model = GRUModel(args.input_size, args.hidden_size, 1, TEXT, 
                     args.dropout, args.bidirectional)
    
    criterion = nn.BCELoss()
    
    for i in range(args.epochs):
        for batch in train_iterator:
            text, text_lengths = batch.text
            # loss = criterion(predictions, batch.label)
        run_eval_loop(val_iterator, model)

def run_eval_loop(val_iterator, model):
    pass

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
                        help='decimal number of percentage of dropout to apply in model')
    parser.add_argument('--bidirectional', type=str2bool,
                        help='True if GRU should be bidirectional, False otherwise')
    
    args = parser.parse_args()
    
    run_training_loop(args)
    
main()
        