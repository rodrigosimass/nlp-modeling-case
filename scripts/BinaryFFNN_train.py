import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import sys
import wandb
import torch.utils.data as data

from TwitterDataset import (
    TwitterDataset_small_train,
    ToToken,
    ToTensor,
)
from BinaryFFNN import BinaryFFNN

TRIAL_RUN = False

if len(sys.argv) < 2:
    print("ERROR! \nusage: python3 MLP.py <<0/1>> for wandb on or off")
    exit(1)
USE_WANDB = bool(int(sys.argv[1]))

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyper-parameters
input_size = 512
hidden_size = 30
num_classes = 1
num_epochs = 50
batch_size = 16
learning_rate = 0.01

# Load datasets of tokenized text
composed = torchvision.transforms.Compose([ToToken(), ToTensor()])

train_dataset = TwitterDataset_small_train(transform=composed)

if TRIAL_RUN:
    train_dataset = data.Subset(train_dataset, range(1500))

train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(
    train_dataset, [train_size, val_size]
)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True
)

val_loader = torch.utils.data.DataLoader(
    dataset=val_dataset, batch_size=batch_size, shuffle=False
)

model = BinaryFFNN(input_size, hidden_size)

criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

if USE_WANDB:
    wandb.init(
        project="nlp-use-case",
        entity="rodrigosimass",
        config={
            "input_size": input_size,
            "hidden_size": hidden_size,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "optimizer": type(optimizer),
            "loss": type(criterion),
        },
    )
    name = "TRIAL_" if TRIAL_RUN else ""
    name += f"FFNN_dropout{hidden_size}"
    wandb.run.name = name

# Early Stopping params
best_val_loss = float("inf")
patience = 3
counter = 0
if USE_WANDB:
    PATH = f"models/{wandb.run.name}.pth"  # store model with name equal to wandb id
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

    with torch.no_grad():
        val_loss = 0
        for messages, labels in val_loader:
            messages = messages.to(device)
            labels = labels.to(device)

            outputs = model(messages)
            val_loss += criterion(outputs, labels).item()

        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            if USE_WANDB:
                torch.save(model.state_dict(), PATH)
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping.")
                break
    if epoch % 1 == 0:
        print(
            f"Epoch [{epoch+1}/{num_epochs}], trn_Loss: {loss.item():.4f}, val_Loss: {val_loss:.4f}"
        )
    if USE_WANDB:
        log_dict = {
            "train_loss": loss.item(),
            "val_loss": val_loss,
        }
        wandb.log(log_dict, step=epoch + 1)


if USE_WANDB:
    wandb.finish()
