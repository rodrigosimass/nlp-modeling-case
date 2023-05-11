import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import sys
import wandb
import torch.utils.data as data

from TwitterDataset import (
    TwitterDataset_small_train,
    TwitterDataset_small_test,
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
hidden_size1 = 20
hidden_size2 = 5
hidden_size3 = 10
num_classes = 1
num_epochs = 30
batch_size = 8
learning_rate = 0.001

num_hidden_layers = 3
dropout = False

# Early Stopping params
patience = 3

# Model creation and parameter logging
model = BinaryFFNN(input_size, hidden_size1, hidden_size2, hidden_size3)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

if USE_WANDB:
    wandb.init(
        project="nlp-use-case",
        entity="rodrigosimass",
        config={
            "input_size": input_size,
            "hidden_size": hidden_size1,
            "hidden_size2": hidden_size2,
            "hidden_size3": hidden_size3,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "optimizer": type(optimizer),
            "loss": type(criterion),
            "num_hidden_layers": num_hidden_layers,
            "dropout_reg": dropout,
            "patience": patience,
        },
    )
    name = "TRIAL_" if TRIAL_RUN else ""
    name += f"FFNN_3layers_small_adam_relu"
    name += "dropout" if dropout else ""
    wandb.run.name = name

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


# Train the model
best_val_loss = float("inf")
counter = 0
for epoch in range(num_epochs):
    model.train()
    trn_loss = 0
    for i, (messages, labels) in enumerate(train_loader):
        messages = messages.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(messages)
        loss = criterion(outputs, labels)
        trn_loss += loss.item()

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    trn_loss /= len(train_loader)

    with torch.no_grad():
        model.eval()
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
                PATH = f"models/{wandb.run.name}.pth"  # store model with name equal to wandb run name
                torch.save(model.state_dict(), PATH)
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping.")
                break
    if epoch % 1 == 0:
        print(
            f"Epoch [{epoch+1}/{num_epochs}], trn_Loss: {trn_loss:.4f}, val_Loss: {val_loss:.4f}"
        )
    if USE_WANDB:
        log_dict = {
            "train_loss": trn_loss,
            "val_loss": val_loss,
        }
        wandb.log(log_dict, step=epoch + 1)

test_dataset = TwitterDataset_small_test(transform=composed)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=test_dataset.n_samples
)

with torch.no_grad():
    n_samples = 0
    n_correct = 0
    model.eval()
    for tokens, labels in test_loader:  # single batch
        outputs = model(tokens)

        predicted_labels = outputs.round()

        n_samples += labels.size(0)
        n_correct += (predicted_labels == labels).sum().item()

        acc = 100.0 * n_correct / n_samples
        print(f"Accuracy of the network on the test set: {acc} %")

if USE_WANDB:
    wandb.finish()
