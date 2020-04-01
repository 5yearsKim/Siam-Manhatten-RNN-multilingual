import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class RNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size=64, num_layers=2, padding_index=0, \
                 embedding_tensor=None):
        super(RNN, self).__init__()
        if torch.is_tensor(embedding_tensor):
            self.encoder = nn.Embedding(vocab_size, embed_size, padding_idx=padding_index, _weight=embedding_tensor)
        else:
            self.encoder = nn.Embedding(vocab_size, embed_size, padding_idx=padding_index)

        self.drop_en = nn.Dropout(p=0.6)

        self.rnn = nn.GRU(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, dropout=0.5,
                          batch_first=True, bidirectional=True)

    def forward(self, x):
        x_embed = self.encoder(x)
        x_embed = self.drop_en(x_embed)

        # None is for initial hidden state
        output, ht = self.rnn(x_embed, None)

        return ht



class Siam_RNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size=64, num_layers=2, padding_index=0, \
             embedding_tensor=None):
        super(Siam_RNN, self).__init__()
        self.rnn = RNN(vocab_size, embed_size, hidden_size, num_layers, padding_index, embedding_tensor)

    def forward(self, sent1, sent2):
        ht1 = self.rnn(sent1)
        ht2 = self.rnn(sent2)
        return ex_neg_man_distance(ht1, ht2)


def ex_neg_man_distance(ht1, ht2):
    prediction = torch.exp(-torch.norm(ht1 - ht2, p=1, dim=[0,2]))
    return prediction


if __name__ == "__main__":
    vocab_size = 10
    embed_size = 20
    model = Siam_RNN(vocab_size, embed_size)
    x = torch.ones(4, 15, dtype=torch.long)
    y = torch.ones(4, 15, dtype=torch.long)
    out = model(x, y)
    print(out)

