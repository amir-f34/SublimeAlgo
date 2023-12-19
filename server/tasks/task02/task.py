import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.jit
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
import os
import os.path
import time
import random
import time
import sys
import json

import numpy as np

np.random.seed(0)
torch.manual_seed(0)
torch.use_deterministic_algorithms(True)

random.seed(6687681968969)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform_to_pil = transforms.ToPILImage()


class GridDataset(Dataset):
    def __init__(self, size=1000):
        tensor_transform = transforms.ToTensor()
        self.filename = "base_image.png"
        img = open(self.filename)

        self.images = list()
        w = 28
        iset = dict()

        img = Image.open(self.filename).convert("RGB")
        for i in range(img.width - w):
            for j in range(img.height - w):
                o = (
                    img.crop((i * w, j * w, i * w + w, j * w + w))
                    .resize((28, 28), Image.NEAREST)
                    .convert("L")
                )
                tr = tensor_transform(o).to(device)

                s = int(torch.sum(tr).tolist() * 1000000)
                if s not in iset:
                    iset[s] = 1
                    self.images.append(tr)
        self.images = self.images[:size]
        tt = len(self.images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]


class DataLoader:
    def __init__(self, batch_size=32, num_workers=0):
        tensor_transform = transforms.ToTensor()
        self.train_data = datasets.MNIST(
            root="./data", train=True, download=True, transform=tensor_transform
        )
        self.loader = torch.utils.data.DataLoader(
            self.train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )


class AutoEncoder(torch.nn.Module):
    def __init__(self, dna_string):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(28 * 28, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 36),
            torch.nn.ReLU(),
            torch.nn.Linear(36, 18),
            torch.nn.ReLU(),
            torch.nn.Linear(18, 9),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(9, 18),
            torch.nn.ReLU(),
            torch.nn.Linear(18, 36),
            torch.nn.ReLU(),
            torch.nn.Linear(36, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 28 * 28),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class Task2:
    def __init__(self, dna_string):
        self.dna_string = dna_string
        self.model = AutoEncoder(dna_string).to(device)
        # Validation using MSE Loss function
        self.lossf = torch.nn.MSELoss()
        # self.lossf = torch.nn.BCELoss()

        self.dataloader = DataLoader()

        # Using an Adam Optimizer with lr = 0.1
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=1e-1, weight_decay=1e-8
        )

    def run(self, epochs=200):
        outputs = []
        losses = []
        for epoch in range(epochs):
            print(epoch)
            for image, _ in self.dataloader.loader:
                image = image.reshape(-1, 28 * 28).to(device)
                reconstructed = self.model(image)
                loss = self.lossf(reconstructed, image)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses.append(loss)
            print(sum(losses) / len(losses))
            outputs.append((epochs, image, reconstructed))
            if epoch % 20 == 0 and epoch > 0:
                transform_to_pil(reconstructed[0].resize(28, 28)).show()


def main(dna_string):
    net = Task2(dna_string)
    ret = net.run(200)
    print(ret)


if __name__ == "__main__":
    main(sys.argv[1])
