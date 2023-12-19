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
import math

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
    def __init__(self, batch_size=5, num_workers=0):
        tensor_transform = transforms.ToTensor()
        self.train_data = datasets.MNIST(
            root="./data", train=True, download=True, transform=tensor_transform
        )

        self.train_data = torch.utils.data.Subset(self.train_data, range(10))
        self.loader = torch.utils.data.DataLoader(
            self.train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )


class AutoEncoderBranching(torch.nn.Module):
    def parse_genes(self, chromosome):
        genes = list()
        i = 0
        while i < len(chromosome):
            char = chromosome[i]
            if char in "abcdefghijk":
                genes.append(char)
                i += 1
            elif char == "s":
                genes.append(chromosome[i : i + 3])
                i += 3
            else:
                i += 1
        return genes

    def __init__(self, dna_string):
        super().__init__()
        self.chromosomes = dna_string.split("|")
        """
        a - linear 0 (identity)
        b - linear 1
        c - linear 2
        d - linear 4
        e - linear 8
        f - linear 16
        g - linear 32
        h - linear 64
        i - linear 128 (8)
        j - linear 256 (9)
        k - linear 512 (10)
        sX1 - split to chromosome X, skip 1
        ##jXa - join to chromosome X, connect to 0
        A - main chromosome
        B - chromosome 2
        C - chromosome 3
        D - ...
          - ...

        mutations: genes: a,b,c,d,e,f,g,h,i,sXy,A,B,C,D...
        mutates may also sp
        
                 e--e        D-chromosome
                 |  |
  x-i--i--i---g--g--g--g- A
    |     |   \       /
    \     /    h-----h       C-chromosome     
     g---g     |     |       B-chromosome
               g-----g       B-chromosome

        A: isBbiigsCcgsDagg
        B: gg
        C: hsBah
        D: ee
        
        isBbiigsCcgsDagg|gg|hsBah|ee
        """
        letter_to_size = {
            "a": 1,
            "b": 2,
            "c": 4,
            "d": 8,
            "e": 16,
            "f": 32,
            "g": 64,
            "h": 128,
            "i": 256,
            "j": 512,
            "k": 1024,
        }
        letter_to_num = {
            "a": 0,
            "b": 1,
            "c": 2,
            "d": 3,
            "e": 4,
            "f": 5,
            "g": 6,
            "h": 7,
            "i": 8,
            "j": 9,
            "k": 10,
        }

        self.chromosomes = dna_string.split("|")
        self.genes = []

        for chromosome in self.chromosomes:
            self.genes.append(self.parse_genes(chromosome))

        lookup = "ABCDEFGHIJKLMNOP"
        self.io_dims = {}
        for i, chromo in enumerate(self.chromosomes):
            chromo_letter = lookup[i]
            if chromo:
                try:
                    self.io_dims[chromo_letter] = [
                        letter_to_size[chromo[0]],
                        letter_to_size[chromo[-1]],
                    ]
                except:
                    self.io_dims[chromo_letter] = [1, 1]
            else:
                self.io_dims[chromo_letter] = [1, 1]
        self.io_dims["A"] = [28 * 28, 16]

        self.layers = []
        for i, chromosome in enumerate(self.genes):
            self.layers.append([])
            in_size = self.io_dims.get(lookup[i], [1])[0]
            current_count = 0
            current_chromosome_sizes = []
            for gene in chromosome:
                if len(gene) == 1:
                    current_chromosome_sizes.append(letter_to_size[gene])

            for gene in chromosome:
                if len(gene) == 1:
                    out_size = letter_to_size[gene]
                    self.layers[-1].append(
                        [
                            "forward",
                            in_size,
                            torch.nn.Linear(in_size, out_size),
                            out_size,
                        ]
                    )
                    in_size = out_size
                    current_count += 1
                else:  # sXx gene:
                    _, chromo, skip = gene
                    skip = letter_to_num[skip]
                    if current_count + skip >= len(current_chromosome_sizes):
                        output = current_chromosome_sizes[-1]
                    else:
                        output = current_chromosome_sizes[current_count + skip]
                    self.layers[-1].append(
                        [
                            "split",
                            lookup.index(chromo),
                            torch.nn.Linear(in_size, self.io_dims[chromo][0]),
                            torch.nn.Linear(self.io_dims[chromo][1], output),
                            current_count + skip,
                        ]
                    )
            if i == 0:
                out_size = self.io_dims["A"][1]
                self.layers[-1].append(
                    ["forward", in_size, torch.nn.Linear(in_size, out_size), out_size]
                )

        # fixed decoder
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(16, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 28 * 28),
            torch.nn.Sigmoid(),
        )

    def run(self, x):
        def run_chromo(num, y):
            memory = []
            for i, data in enumerate(self.layers[num]):
                if data[0] == "forward":
                    y = data[2](y)
                elif data[0] == "split":
                    z = run_chromo(data[1], data[2](y))
                    z = data[3](z)
                    memory.append([data[4], data, z])
                for merge, data, value in memory:
                    if merge == i + 1:
                        y = y + value
            return y

        return run_chromo(0, x)

    def forward(self, x):
        encoded = self.run(x)
        decoded = self.decoder(encoded)
        return decoded


class Task5:
    def __init__(self, dna_string):
        self.dna_string = dna_string
        self.model = AutoEncoderBranching(dna_string).to(device)
        self.lossf = torch.nn.MSELoss()
        self.dataloader = DataLoader()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=1e-1, weight_decay=1e-8
        )

    def run(self, epochs=100):
        if len(self.model.layers) == 0 or len(self.model.layers[0]) == 0:
            return 999999999
        try:
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
            max_len = max([len(x) for x in self.dna_string.split("|")])
            out_loss = out_loss * (1 + (max_len**2) / 128)
            base_layers = [x[1] for x in self.model.layers[0] if x[0] == "forward"]
            depth = len(base_layers)
            avg_wide = math.log(sum(base_layers) / depth)
            out_loss = 500 * out_loss / ((5 + depth) * (1 + avg_wide))
            return out_loss
        except RecursionError:
            return 999999999


def main(dna_string):
    net = Task5(dna_string)
    ret = net.run(200)
    print(ret)


if __name__ == "__main__":
    main(sys.argv[1])
