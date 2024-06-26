from collections import Counter
import datetime
import getpass
import os

from pathlib import Path
from PIL import Image
import torch
import numpy as np

import sklearn
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import transforms, datasets
import pytorch_lightning as pl
import torch.multiprocessing as mp

from torch.utils.data import ConcatDataset, random_split

class MixtecGenders(pl.LightningDataModule):
    def __init__(self, data_dir=None, batch_size=125, num_workers=8, input_transforms=None, category=""):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.category = str(category)

        self.reference_dataloader = None

        if data_dir is None:
            self.basepath = Path(
                f"/home/{getpass.getuser()}/toorange/alexwebber/mixteclabeling"
            )  # Base data directory
            self.data_dir = self.basepath
        else:
            self.data_dir = data_dir

        ## Set default transforms, then append based on input transform array
        self.input_transforms = transforms.Compose(
                        [
                            transforms.ToTensor(),
                            transforms.Resize((224, 224), antialias=True),
                        ]
                    )


        if 'RandomErasing' in input_transforms:
            print("Using random erasing")
            self.input_transforms.transforms.append(transforms.RandomErasing())

        if 'RandomHorizontalFlip' in input_transforms:
            print("Using random horizontal flip")
            self.input_transforms.transforms.append(transforms.RandomHorizontalFlip())

        if 'RandomVerticalFlip' in input_transforms:
            print("Using random vertical flip")
            self.input_transforms.transforms.append(transforms.RandomVerticalFlip())

    def prepare_dataset(self):
        string_path_v = "data/labeled_figures/codex_vindobonensis/" + self.category + "/"
        string_path_n = "data/labeled_figures/codex_nuttall/" + self.category + "/"
        string_path_s = "data/labeled_figures/codex_selden/" + self.category + "/"


        self.path_v = self.basepath / string_path_v
        self.path_n = self.basepath / string_path_n
        self.path_s = self.basepath / string_path_s

    def setup(self, stage):
        ## Load images into PyTorch dataset
        transform = self.input_transforms

        self.prepare_dataset()

        vindobonensis_dataset = datasets.ImageFolder(self.path_v, transform=transform)
        nuttall_dataset = datasets.ImageFolder(self.path_n, transform=transform)
        selden_dataset = datasets.ImageFolder(self.path_s, transform=transform)

        self.figures_dataset = ConcatDataset(
            [vindobonensis_dataset, nuttall_dataset, selden_dataset]
        )

        self.train_set, self.val_set, self.test_set = random_split(
            self.figures_dataset, [0.6, 0.2, 0.2]
        )
        
        train_labels = [item[1] for item in self.train_set]
        print(f"Training Set: {Counter(train_labels)}")

        val_labels = [item[1] for item in self.val_set]
        print(f"Validation Set: {Counter(val_labels)}")

        test_labels = [item[1] for item in self.test_set]
        print(f"Test Set: {Counter(test_labels)}")
    

    @staticmethod
    def get_reference_dataloader():
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((224, 224), antialias=True),
            ]
        )

        refimageset = datasets.ImageFolder("../reference_images/", transform=transform)
        return DataLoader(refimageset, batch_size=1)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

def createConfusionMatrix(loader, net):
    y_pred = [] # save predction
    y_true = [] # save ground truth

    # iterate over data
    for inputs, labels in loader:
        output = net(inputs)  # Feed Network

        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output)  # save prediction

        labels = labels.data.cpu().numpy()
        y_true.extend(labels)  # save ground truth

    # constant for classes
    classes = ("female", "male")

    # Build confusion matrix
    cf_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix, index=[i for i in classes],
                         columns=[i for i in classes])
    plt.figure(figsize=(12, 7))    
    return sn.heatmap(df_cm, annot=True).get_figure()