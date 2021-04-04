import torch
import torch.backends.cudnn
import pandas as pd

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


def read_csv(file, train_val_split=False):
    df = pd.read_csv(file)

    df = df[["stemmed", "Party"]]
    df.loc[df["Party"] == "Independent", "Party"] = "Democrat"

    le = LabelEncoder()
    df["Party"] = le.fit_transform(df["Party"].values)

    train, val = train_test_split(df, test_size=0.1)
    if train_val_split:
        return train, val
    else:
        return df, val


def get_data_iterator(train, val, fields, batch_size, device):
    train_ds = DataFrameDataset(train, fields)
    val_ds = DataFrameDataset(val, fields)
    iterator = data.BucketIterator.splits(
        (train_ds, val_ds),
        batch_size=batch_size,
        device=device,
        sort_key=lambda x: len(x.text),
        sort=False,
        shuffle=True,
        sort_within_batch=True,
    )
    return iterator


def create_fields():
    TEXT = data.Field(tokenize="spacy")
    LABEL = data.LabelField(dtype=torch.float)

    fields = [("text", TEXT), ("label", LABEL)]
    return fields, TEXT, LABEL


def set_vocab(text, label, dataset, max_vocab_size):
    text.build_vocab(dataset, max_size=max_vocab_size,
                     vectors="glove.6B.200d", unk_init=torch.Tensor.zero_)
    label.build_vocab(dataset)


def setup_fields(dataframe, max_vocab_size):

    fields, TEXT, LABEL = create_fields()

    train_ds = DataFrameDataset(dataframe[0], fields)
    set_vocab(TEXT, LABEL, train_ds, max_vocab_size)

    return fields, TEXT, LABEL
