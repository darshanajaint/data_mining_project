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

    def __len__(self):
        return len(self.examples)


def read_csv(file, train_val_split=False):
    df = pd.read_csv(file)

    df = df[["stemmed", "Party"]]
    df.loc[df["Party"] == "Independent", "Party"] = "Democrat"

    le = LabelEncoder()
    df["Party"] = le.fit_transform(df["Party"].values)

    if train_val_split:
        return train_test_split(df, test_size=0.1)
    else:
        return df


def get_data_iterator(dataframe, fields, batch_size, device):
    dataset = DataFrameDataset(dataframe, fields)
    iterator = data.BucketIterator.splits(
        dataset,
        batch_size=batch_size,
        device=device,
        sort_key=lambda x: len(x.text),
        sort=False,
        shuffle=True,
        sort_within_batch=True,
    )
    return iterator, dataset


def create_fields():
    SEED = 0
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    TEXT = data.Field(tokenize="spacy")
    LABEL = data.LabelField(dtype=torch.float)

    fields = [("text", TEXT), ("label", LABEL)]
    return fields, TEXT, LABEL


def set_vocab(text, label, dataset, max_vocab_size):
    text.build_vocab(dataset, max_size=max_vocab_size,
                     vectors="glove.6B.200d", unk_init=torch.Tensor.zero_)
    label.build_vocab(dataset)


def setup_fields(file_name, max_vocab_size):
    train_df, _ = read_csv(file_name, True)

    fields, TEXT, LABEL = create_fields()

    train_ds = DataFrameDataset(train_df, fields)
    set_vocab(TEXT, LABEL, train_ds, max_vocab_size)

    return fields, TEXT, LABEL
