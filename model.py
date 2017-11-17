import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split



def get_data():
    X = np.random.rand(200, 10)  # in the form nsamples X nParameters (redshift, source_type, etc.)
    y = np.random.rand(200, 5)  # here nsamples X nBetas

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=args.seed)

    return X_train, X_test, y_train, y_test


class RTnet(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 100)
        self.fc2 = nn.Linear(100, 200)
        self.fc3 = nn.Linear(200, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # No non-linearity here
        return x


model = RTnet()

if args.cuda:
    model.cuda()

# Define the optimizer

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

def train(epoch, X_train, y_train):
    model.train()

    nbatches = X_train.shape[0]//args.batch_size

    for batch_idx in range(nbatches):
        start = batch_idx*nbatches
        end = start + nbatches

        data = X_train[start:end,:]
        target = y_train[start:end, :]

        if args.cuda:
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, target)
        loss.backward()

def run(args):
    X_train, X_test, y_train, y_test = get_data()
    print("Size of X is {}".format(X.shape))

    X_train = torch.from_numpy(X_train)
    X_test = torch.from_numpy(X_test)
    y_train = torch.from_numpy(y_train)
    y_test = torch.from_numpy(y_test)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MLP for Radiative Transfer')

    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    run(args)