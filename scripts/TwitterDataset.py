import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel
import torchvision

import tensorflow_hub as hub
import tensorflow_text

module_url = "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"
model = hub.load(module_url)


def encode_text(text):
    return model(text)


class TwitterDataset_small_train(Dataset):
    def __init__(self, transform=None) -> None:
        df = pd.read_csv(
            "data/processed/dataset_small_w_bart_preds_and_original_message.csv",
            delimiter=",",
        )

        df = df[:15000]

        self.n_samples = df.shape[0]

        self.x = df.message
        self.y = df.is_positive

        self.transform = transform

    def __getitem__(self, index):
        sample = self.x[index], self.y[index]

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.n_samples

class TwitterDataset_small_test(Dataset):
    def __init__(self, transform=None) -> None:
        df = pd.read_csv(
            "data/processed/dataset_small_w_bart_preds_and_original_message.csv",
            delimiter=",",
        )

        df = df[-5000:]
        df = df.reset_index(drop=True)


        self.n_samples = df.shape[0]

        self.x = df.message
        self.y = df.is_positive

        self.transform = transform

    def __getitem__(self, index):
        sample = self.x[index], self.y[index]

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.n_samples

class ToToken:
    def __call__(self, sample):
        message, label = sample

        tokens = encode_text(message)
        return tokens, label


class ToTensor:
    def __call__(self, sample):
        tokens, label = sample
        tokens =  torch.from_numpy(tokens.numpy().astype(np.float32))
        label = torch.tensor(label.astype(np.float32)).reshape(-1,1)
        return tokens, label


if __name__ == "__main__":
    # create dataset
    composed = torchvision.transforms.Compose([ToToken(), ToTensor()])
    
    dataset_train = TwitterDataset_small_train(transform=composed)

    first_data = dataset_train[0]
    features, label = first_data
    print(type(features), type(label))
    print(features.shape, label.shape)
    print(dataset_train.n_samples)

    dataset_test = TwitterDataset_small_test(transform=composed)

    first_data = dataset_test[0]
    features, label = first_data
    print(type(features), type(label))
    print(features.shape, label.shape)
    print(dataset_test.n_samples)
