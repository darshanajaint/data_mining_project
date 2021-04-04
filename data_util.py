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
            print(self.examples)
            break
        super().__init__(self.examples, fields)

    def __getitem__(self, i):
        return self.examples[i]


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
    return data.BucketIterator.splits(
        dataset,
        batch_size=batch_size,
        device=device,
        sort_key=lambda x: len(x.text),
        sort=False,
        shuffle=True,
        sort_within_batch=True,
    ), dataset


def create_fields():
    SEED = 0
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    TEXT = data.Field(tokenize="spacy")
    LABEL = data.LabelField(dtype=torch.float)

    fields = [("text", TEXT), ("label", LABEL)]
    return fields, TEXT, LABEL


def set_vocab(text, label, build_vocab=False, dataset=None, max_vocab_size=0,
              text_vocab_path='./text_vocab.txt',
              label_vocab_path='./label_vocab.txt'):
    if build_vocab:
        text.build_vocab(dataset, max_size=max_vocab_size,
                         vectors="glove.6B.200d", unk_init=torch.Tensor.zero_)
        label.build_vocab(dataset)

        save_vocab(text.vocab, text_vocab_path)
        save_vocab(label.vocab, label_vocab_path)
    else:
        load_vocab(text_vocab_path, text)
        load_vocab(label_vocab_path, label)


def setup_fields(file_name, max_vocab_size, build_vocab, text_vocab_path,
                 label_vocab_path):
    train_df, _ = read_csv(file_name, True)

    fields, TEXT, LABEL = create_fields()

    if build_vocab:
        train_ds = DataFrameDataset(train_df, fields)
        set_vocab(TEXT, LABEL, build_vocab, train_ds, max_vocab_size,
                  text_vocab_path, label_vocab_path)
        del train_ds
    else:
        print("Through here")
        set_vocab(TEXT, LABEL, build_vocab=build_vocab,
                  text_vocab_path=text_vocab_path,
                  label_vocab_path=label_vocab_path)

    del train_df
    return fields, TEXT, LABEL


def save_vocab(vocab, path):
    with open(path, "w+", encoding="utf-8") as output:
        for token, index in vocab.stoi.items():
            output.write(f'{index}\t{token}\n')


def load_vocab(path, field):
    vocab = dict()
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            index, token = line.split("\t")
            vocab[token] = int(index)
        field.vocab = vocab
