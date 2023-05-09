import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import wandb

import sys

sys.path.append("models")

from TwitterDataset import (
    TwitterDataset_small_train,
    TwitterDataset_small_test,
    ToToken,
    ToTensor,
)
from BinaryFFNN import BinaryFFNN


# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyper-parameters
input_size = 512
hidden_size = 200
num_classes = 1
num_epochs = 10
batch_size = 32
learning_rate = 0.001

# Load datasets of tokenized text
composed = torchvision.transforms.Compose([ToToken(), ToTensor()])

train_dataset = TwitterDataset_small_train(transform=composed)

x = train_dataset.x
y = train_dataset.y

print(x.shape)
print(y.shape)

test_dataset = TwitterDataset_small_test(transform=composed)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=False
)

model = BinaryFFNN(input_size, hidden_size)

criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Train the model
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (messages, labels) in enumerate(train_loader):

        messages = messages.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(messages)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 500 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}"
            )

print('Finished Training')
PATH = 'models/ffnn.pth'
torch.save(model.state_dict(), PATH)

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for tokens, labels in test_loader:
        tokens = tokens.to(device)
        labels = labels.to(device)
        outputs = model(tokens)
        # max returns (value ,index)
        predicted_labels = outputs.round()
       
        n_samples += labels.size(0)
        n_correct += (predicted_labels == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f"Accuracy of the network on the test set: {acc} %")
