import torch
import torch.backends.cudnn
import pandas as pd
import torchtext.data as data
import pickle

from torchtext.legacy import data

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


class DataFrameDataset(data.Dataset):
    def __init__(self, df, fields):
        self.examples = []
        for i, row in df.iterrows():
            label = row["Party"]
            text = row["stemmed"]
            self.examples.append(data.Example.fromlist([text, label], fields))
        super().__init__(self.examples, fields)

    def __getitem__(self, i):
        return self.examples[i]


def load_data(file_name, max_vocab_size, batch_size, device, build_vocab=True,
              text_vocab_path="./text_vocab.pickle",
              label_vocab_path="./label_vocab.pickle"):
    train_df, valid_df = read_csv(file_name)

    SEED = 0
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    TEXT = data.Field(tokenize="spacy")
    LABEL = data.LabelField(dtype=torch.float)
    fields = [("text", TEXT), ("label", LABEL)]

    train_ds = DataFrameDataset(train_df, fields)
    val_ds = DataFrameDataset(valid_df, fields)

    if build_vocab:
        TEXT.build_vocab(train_ds, max_size=max_vocab_size,
                         vectors="glove.6B.200d",
                         unk_init=torch.Tensor.zero_)
        LABEL.build_vocab(train_ds)

        save_vocab(TEXT.vocab, text_vocab_path)
        save_vocab(LABEL.vocab, label_vocab_path)
    else:
        load_vocab(text_vocab_path, TEXT)
        load_vocab(label_vocab_path, LABEL)

    train_iterator, val_iterator = data.BucketIterator.splits(
        # Datasets for iterator to draw data from
        (train_ds, val_ds),

        # Tuple of train and validation batch sizes.
        batch_sizes=(batch_size, batch_size),

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

    return train_iterator, val_iterator, TEXT, LABEL


def read_csv(file):
    df = pd.read_csv(file)

    df = df[["stemmed", "Party"]]
    df.loc[df["Party"] == "Independent", "Party"] = "Democrat"

    le = LabelEncoder()
    df["Party"] = le.fit_transform(df["Party"].values)

    return train_test_split(df, test_size=0.1)


def save_vocab(vocab, path):
    with open(path, "wb") as output:
        pickle.dump(vocab, output)


def load_vocab(path, field):
    with open(path, "r") as file:
        field.vocab = pickle.load(file)