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

        self.train_data = torch.utils.data.Subset(self.train_data, range(100))
        self.loader = torch.utils.data.DataLoader(
            self.train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )


class AutoEncoderActivation(torch.nn.Module):
    def __init__(self, dna_string):
        super().__init__()
        self.segs = []
        for chromosome in dna_string.split("|"):
            self.segs.append([])
            for i, x in enumerate(chromosome):
                if i % 2 == 0:
                    self.segs[-1].append(x)
                else:
                    self.segs[-1][-1] += x

        out_s = list()
        for c in self.segs:
            out_s.append([])
            for seg in c:
                out_s[-1].append(int(seg, 16))
        self.out_s = out_s
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(28 * 28, 128),
            Act(out_s),
            torch.nn.Linear(128, 64),
            Act(out_s),
            torch.nn.Linear(64, 36),
            Act(out_s),
            torch.nn.Linear(36, 18),
            Act(out_s),
            torch.nn.Linear(18, 9),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(9, 18),
            Act(out_s),
            torch.nn.Linear(18, 36),
            Act(out_s),
            torch.nn.Linear(36, 64),
            Act(out_s),
            torch.nn.Linear(64, 128),
            Act(out_s),
            torch.nn.Linear(128, 28 * 28),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class Act(nn.Module):
    def __init__(self, out_s):
        super().__init__()
        self.out_s = out_s

    def forward(self, x):
        div1 = self.out_s[0][0]
        output1 = torch.zeros(x.size())
        for i, num in enumerate(self.out_s[0][1:]):
            output1 += (num / div1) * torch.pow(x, 1 / (i + 1))  # pow(x,1/(i+1))

        div2 = self.out_s[0][0]
        output2 = torch.zeros(x.size())
        for i, num in enumerate(self.out_s[1][1:]):
            output2 += (num / div2) * torch.pow(x, 1 / (i + 1))  # pow(-x, 1/(i+1))

        output = torch.where(x > 0, output1, output2)

        return output


class Task4:
    def __init__(self, dna_string):
        self.dna_string = dna_string
        self.model = AutoEncoderActivation(dna_string).to(device)
        self.lossf = torch.nn.MSELoss()

        self.dataloader = DataLoader()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=1e-1, weight_decay=1e-8
        )

    def run(self, epochs=20):
        if self.dna_string.count("|") > 1:
            return 9999999999
        if len(self.dna_string) > 32 or len(self.dna_string) < 6:
            return 999999999
        if len(self.dna_string.split("|")) < 2:
            return 999999999
        if len(self.dna_string.split("|")[0]) < 4:
            return 999999999
        if len(self.dna_string.split("|")[1]) < 4:
            return 999999999
        for x in self.dna_string.split("|"):
            if not x or int(x[:2], 16) == 0:
                return 999999999

        outputs = []
        losses = []
        out_loss = 999999999
        for epoch in range(epochs):
            for image, _ in self.dataloader.loader:
                image = image.reshape(-1, 28 * 28).to(device)

                reconstructed = self.model(image)

                loss = self.lossf(reconstructed, image)
                if loss.isnan():
                    return 999999999
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                losses.append(loss.tolist())
            out_loss = sum(losses) / len(losses)
            outputs.append((epochs, image, reconstructed))
        return out_loss * (1 - len(self.dna_string) / 128)


def main(dna_string):
    net = Task4(dna_string)
    ret = net.run(200)
    print(ret)


if __name__ == "__main__":
    main(sys.argv[1])
