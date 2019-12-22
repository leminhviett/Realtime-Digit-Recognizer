import torch
import torch.nn as nn
import pandas as pd
import torch.optim as optim


from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader


class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(32 * 4 * 4, 10)

    def forward(self, x):
        # Convolution 1
        out = self.cnn1(x)
        out = self.relu1(out)

        # Max pool 1
        out = self.maxpool1(out)

        # Convolution 2
        out = self.cnn2(out)
        out = self.relu2(out)

        # Max pool 2
        out = self.maxpool2(out)
        out = out.view(out.size(0), -1)

        # Linear function (readout)
        out = self.fc1(out)

        return out

if __name__ == '__main__':
    train = pd.read_csv('data/train.csv')

    X_numpy = train.loc[:, train.columns != 'label'].values / 255
    y_numpy = train.loc[:, 'label'].values
    print("X_train: {}. y_train {}".format(X_numpy.shape, y_numpy.shape))

    X_train_all = torch.from_numpy(X_numpy).type(torch.LongTensor)
    y_train_all = torch.from_numpy(y_numpy)
    train_all = TensorDataset(X_train_all, y_train_all)
    train_loader_all = DataLoader(train_all, batch_size=100, shuffle=False)

    model = CNNModel()
    optimizer = torch.optim.Adagrad(model.parameters(), lr = 0.1)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(15):
      for i, (imgs, labels) in enumerate(train_loader_all):

        train = imgs.view(100, 1, 28, 28).type(torch.FloatTensor)
        labels = labels

        optimizer.zero_grad()
        outputs = model(train)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        if(i%100 == 0):
          print("epoch = {}, i = {}, loss = {}".format(epoch, i, loss))


    torch.save(model, 'digitRecognizer.pt') #save model