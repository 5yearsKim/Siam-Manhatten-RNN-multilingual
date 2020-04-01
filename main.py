from __future__ import print_function

import json
import argparse

import random
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataloader import pair_dataset
from model import Siam_RNN
from utils import AverageMeter, AccuracyMeter
from utils import adjust_learning_rate


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--epochs', default=30, type=int, help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=16, type=int, help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.005, type=float, help='initial learning rate')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, help='weight decay')
parser.add_argument('--print-freq', '-p', default=50, type=int, help='print frequency')
parser.add_argument('--save-freq', '-sf', default=1, type=int, help='model save frequency(epoch)')
parser.add_argument('--embedding-size', default=128, type=int, help='embedding size')
parser.add_argument('--seq_len', default=20, type=int, help='sequence length')
parser.add_argument('--hidden-size', default=128, type=int, help='rnn hidden size')
parser.add_argument('--layers', default=2, type=int, help='number of rnn layers')
parser.add_argument('--min-samples', default=5, type=int, help='min number of tokens')
parser.add_argument('--clip', type=float, default=0.25, help='gradient clipping')
parser.add_argument('--resume', type=bool, default=False, help='resume from checkpoint')
parser.add_argument('--load_path', type=str, default="ckpts/rnn_11.pt", help='load path')
args = parser.parse_args()


word2index = json.load( open( "./dataset/word2index.json" ) )

# create trainer
print("===> creating dataloaders ...")
testset_size = 1000
pair_dset = pair_dataset(args.seq_len, path='./dataset/XNLI-15way/en_es_siam_processed.csv')
train_dset, test_dset = torch.utils.data.random_split(pair_dset, [len(pair_dset) - testset_size, testset_size])
train_loader = DataLoader(train_dset, batch_size=args.batch_size)
test_loader = DataLoader(test_dset, batch_size=args.batch_size)

# create model
print("===> creating rnn model ...")
device = torch.device( "cuda:0" if torch.cuda.is_available() else "cpu" )
vocab_size = len(word2index) + 3
model = Siam_RNN(vocab_size=vocab_size, embed_size=args.embedding_size,
            hidden_size=args.hidden_size, embedding_tensor=None, num_layers=args.layers).to(device)
print(model)
if args.resume:
    model.load_state_dict(torch.load(args.load_path))
# optimizer and loss
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

mseloss = torch.nn.MSELoss()

def train_pair(pair_loader, model, optimizer, device):

    # switch to train mode
    model.train()
    tot_loss = AverageMeter()
    for i, (en, es, label) in enumerate(pair_loader):
        # compute output
        pred = model(en.to(device), es.to(device))
        loss = mseloss(pred, label.type(torch.float).to(device))
        tot_loss.update(loss.item(), pred.shape[0])

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        if i != 0 and i % args.print_freq == 0:
            print('Iteration {} -> Train Loss: {:.4f}'.format(i, tot_loss.avg))
            tot_loss.reset()




def test(pair_loader, model, device):
    acc_meter = AccuracyMeter()

    # switch to evaluate mode
    model.eval()
    for i, (en, es, label) in enumerate(pair_loader):

        # compute output
        pred = model(en.to(device), es.to(device))
        pred_val = pred.view(-1).ge(0.5).type(torch.long)
        hit = pred_val[torch.eq(pred_val, label)].shape[0]
        acc_meter.update(hit, pred.shape[0])
    print('Accuracy {:.4f} % '.format(acc_meter.accuracy * 100))


# training and testing
for epoch in range(1, args.epochs+1):
    print(epoch, "th epochs running")
    adjust_learning_rate(args.lr, optimizer, epoch )
    train_pair(train_loader, model, optimizer, device)
    test(test_loader, model, device)
    # save current model
    if epoch % args.save_freq == 0:
        path_model = './ckpts/siam_{}.pt'.format(epoch)
        torch.save(model.state_dict(), path_model)
        print(path_model, " is saved successfully!")

