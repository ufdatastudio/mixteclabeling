from collections import Counter
import datetime
import getpass
import os

from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import pytorch_lightning as pl
import torch.multiprocessing as mp

from torch.utils.data import ConcatDataset, random_split

class MixtecGenders(pl.LightningDataModule):
    def __init__(self, data_dir=None, batch_size=125, num_workers=8):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.reference_dataloader = None

        if data_dir is None:
            self.basepath = Path(
                f"/home/{getpass.getuser()}/toorange/alexwebber/mixteclabeling"
            )  # Base data directory
            self.data_dir = self.basepath
        else:
            self.data_dir = data_dir

    def prepare_dataset(self):
        self.path_v = self.basepath / "data/labeled_figures/codex_vindobonensis/gender/"
        self.path_n = self.basepath / "data/labeled_figures/codex_nuttall/gender/"
        self.path_s = self.basepath / "data/labeled_figures/codex_selden/gender/"

    def setup(self, stage):
        ## Load images into PyTorch dataset
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # AddRandomBlockNoise(),
                transforms.Resize((224, 224), antialias=True),
                # transforms.Grayscale(),
                # transforms.ColorJitter(contrast=0.5),
                # transforms.RandomRotation(360),     # Maybe useful for standng and sitting
                # transforms.RandomHorizontalFlip(50),
                # transforms.RandomVerticalFlip(50)
            ]
        )

        self.prepare_dataset()

        vindobonensis_dataset = datasets.ImageFolder(self.path_v, transform=transform)
        nuttall_dataset = datasets.ImageFolder(self.path_n, transform=transform)
        selden_dataset = datasets.ImageFolder(self.path_s, transform=transform)

        self.figures_dataset = ConcatDataset(
            [vindobonensis_dataset, nuttall_dataset, selden_dataset]
        )

        self.train_set, self.val_set, self.test_set = random_split(
            self.figures_dataset, [0.6, 0.3, 0.1]
        )

        print(dict(Counter(vindobonensis_dataset.targets)))
    

    @staticmethod
    def get_reference_dataloader():
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # AddRandomBlockNoise(),
                transforms.Resize((224, 224), antialias=True),
                # transforms.Grayscale(),
                # transforms.ColorJitter(contrast=0.5),
                # transforms.RandomRotation(360),     # Maybe useful for standng and sitting
                # transforms.RandomHorizontalFlip(50),
                # transforms.RandomVerticalFlip(50)
            ]
        )

        refimageset = datasets.ImageFolder("reference_images/", transform=transform)
        return DataLoader(refimageset, batch_size=1)


    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
