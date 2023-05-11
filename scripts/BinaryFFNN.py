import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F

class BinaryFFNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3):
        super(BinaryFFNN, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size1) 
        self.l2 = nn.Linear(hidden_size1, hidden_size2)
        #self.l3 = nn.Linear(hidden_size2, hidden_size3)  
        self.l4 = nn.Linear(hidden_size2, 1)
        self.relu = nn.ReLU()
        self.lrelu = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(p=0.5)
    
    def forward(self, x):
        x = self.relu(self.l1(x))
        x = self.dropout(x)
        x = self.relu(self.l2(x))
        x = self.dropout(x)
        #x = self.relu(self.l3(x))
        #x = self.dropout(x)
        x = self.l4(x)
        x = torch.sigmoid(x)
        # sigmoid is required in binary classifier
        return x