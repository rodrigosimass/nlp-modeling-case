import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F

class BinaryFFNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BinaryFFNN, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(p=0.5)
    
    def forward(self, x):
        x = self.relu(self.l1(x))
        x = self.dropout(x)
        x = self.l2(x)
        x = torch.sigmoid(x)
        # sigmoid is required in binary classifier
        return x