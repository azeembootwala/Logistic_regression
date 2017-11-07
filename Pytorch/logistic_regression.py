import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from util import get_data
from torch.autograd import Variable


class LogisticRegression(nn.Module):
    def __init__(self, in_dim , out_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(in_dim , out_dim)

    def forward(self, X):
        return self.linear(X)

def train(model , loss , optimizer, inputs , labels):
    if torch.cuda.is_available():
        inputs = Variable(inputs.cuda())
        labels = Variable(labels.cuda())
    else:
        inputs = Variable(inputs)
        labels=Variable(labels)

    optimizer.zero_grad()

    output = model(inputs)

    cost = loss(output, labels)

    cost.backward()

    optimizer.step()


def predict(model,inputs, labels):
    inputs = Variable(inputs.cuda())
    out = model(inputs)
    _,pred =torch.max(out.data, 1)
    return pred


def main():
    X, Y = get_data("train")
    Xtrain = X[:-1000]
    Ytrain = Y[:-1000]
    Xtest = X[-1000:]
    Ytest = Y[-1000:]

    Xtrain = torch.from_numpy(Xtrain).float()
    Ytrain = torch.from_numpy(Ytrain).long()
    Xtest  = torch.from_numpy(Xtest).float()

    in_dim = Xtrain.size(1)
    out_dim = 10
    model = LogisticRegression(in_dim, out_dim)
    if torch.cuda.is_available():
        model.cuda()

    loss = nn.CrossEntropyLoss()

    learning_rate = 0.001

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    max_iter = 300
    batch_size=100
    n_batches = int(Xtrain.size(1)/batch_size)


    for i in range(0,max_iter):

        for j in range(0,n_batches):
            Xbatch = Xtrain[j*batch_size:(j+1)*batch_size]
            Ybatch = Ytrain[j*batch_size:(j+1)*batch_size]

            train(model , loss , optimizer , Xbatch, Ybatch)


            if j % 20 ==0:

                pred=predict(model,Xtest, Ytest)

                acc = np.mean(pred.cpu().numpy()==Ytest)



if __name__ =="__main__":
    main()
