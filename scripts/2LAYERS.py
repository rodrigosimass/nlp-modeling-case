import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F


class BinaryFFNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2=None, use_dropout=False):
        super(BinaryFFNN, self).__init__()
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.use_dropout = use_dropout

        self.l1 = nn.Linear(input_size, hidden_size1)
        self.relu = nn.ReLU()

        if hidden_size2:
            self.l2 = nn.Linear(hidden_size1, hidden_size2)
            self.l3 = nn.Linear(hidden_size2, 1)
        else:
            self.l3 = nn.Linear(hidden_size2, 1)

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.relu(self.l1(x))
        
        if self.use_dropout:
            x = self.dropout(x)

        if self.hidden_size2:
            x = self.relu(self.l2(x))

        if self.use_dropout:
            x = self.dropout(x)

        x = self.l2(x)
        x = torch.sigmoid(x)
        # sigmoid is required in binary classifier
        return x
