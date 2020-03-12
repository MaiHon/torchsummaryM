from torchsummaryM import summary

import numpy as np
import torchvision
import torch
from torch import nn
import torch.nn.functional as F

def test_conv():
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
            self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
            self.conv2_drop = nn.Dropout2d()
            self.fc1 = nn.Linear(320, 50)
            self.fc2 = nn.Linear(50, 10)

        def forward(self, x):
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
            x = x.view(-1, 320)
            x = F.relu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)
    summary_value, _ = summary(Net(), torch.zeros((1, 1, 28, 28)))

def test_resnet_50():
    model = torchvision.models.resnet50()
    summary_value, _ = summary(model, torch.zeros((4, 3, 224, 224)))

def test_lstm():
    class Net(nn.Module):
        def __init__(self,
                    vocab_size=20, embed_dim=300,
                    hidden_dim=512, num_layers=2):
            super().__init__()

            self.hidden_dim = hidden_dim
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            self.encoder = nn.LSTM(embed_dim, hidden_dim,
                                num_layers=num_layers)
            self.decoder = nn.Linear(hidden_dim, vocab_size)

        def forward(self, x):
            embed = self.embedding(x)
            out, hidden = self.encoder(embed)
            out = self.decoder(out)
            out = out.view(-1, out.size(2))
            return out, hidden
    inp = torch.zeros((1, 100)).long() # [length, batch_size]
    df, df_total = summary(Net(), inp)

def test_multi_inputs():
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1_x = nn.Conv2d(1, 10, kernel_size=5)
            self.conv2_x = nn.Conv2d(10, 20, kernel_size=5)
            self.conv2_drop_x = nn.Dropout2d()

            self.conv1_y = nn.Conv2d(1, 10, kernel_size=5)
            self.conv2_y = nn.Conv2d(10, 20, kernel_size=5)
            self.conv2_drop_y = nn.Dropout2d()

            self.fc1 = nn.Linear(320, 50)
            self.fc2 = nn.Linear(50, 10)

        def forward(self, x, y):
            y = F.relu(F.max_pool2d(self.conv1_y(y), 2))
            y = F.relu(F.max_pool2d(self.conv2_drop_y(self.conv2_y(y)), 2))

            x = F.relu(F.max_pool2d(self.conv1_x(x), 2))
            x = F.relu(F.max_pool2d(self.conv2_drop_x(self.conv2_x(x)), 2))
            x = x.view(-1, 320)
            x = F.relu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)
    
    inp1 = torch.zeros((1, 1, 28, 28))
    inp2 = torch.zeros((1, 1, 64, 64))
    summary_value, _ = summary(Net(), inp1, inp2)

if __name__ == "__main__":
    test_conv()
    test_lstm()
    test_resnet_50()
    test_multi_inputs()
    