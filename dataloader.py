import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader

PAD = 0
SOS = 1
EOS = 2

class pair_dataset(Dataset):
    def __init__(self, seq_len, path="./dataset/XNLI-15way/en_es_siam_processed.csv"):
        self.df = pd.read_csv(path, sep=',', index_col=0)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        en_sent = torch.tensor(add_padding(self.df.loc[idx, 'en'], self.seq_len))
        es_sent = torch.tensor(add_padding(self.df.loc[idx, 'es'], self.seq_len))
        label = torch.tensor(self.df.loc[idx, 'label'])
        return en_sent, es_sent, label


def add_padding(sent, seq_len):
    sent = list(map(lambda x: int(x) + 3, sent.split()))
    if len(sent) >= seq_len -2:
        sent = sent[:seq_len-2]
    sent = [SOS] + sent + [EOS] + [PAD] * (seq_len - 2 - len(sent))
    return sent


if __name__ == "__main__":
    dset = pair_dataset(30)
    dloader = DataLoader(dset, batch_size=4)
    print(next(iter(dloader))[3])
