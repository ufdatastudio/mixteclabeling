import datetime
import getpass
import os

from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader
import pytorch_lightning as pl


class MixtecGenders(pl.LighteningDataModule):
    def __init__(self, data_dir=None, batch_size=125, num_workers=None):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers if num_workers is not None else "auto"

        if data_dir is None:
            basepath = Path(
                f"/home/{getpass.getuser()}/toorange/alexwebber/mixteclabeling"
            )  # Base data directory
            self.data_dir = basepath
        else:
            self.data_dir = data_dir

    def prepare_dataset(self):
        self.path_v = self.basepath / "data/labeled_figures/codex_vindobonensis/gender/"
        self.path_n = self.basepath / "data/labeled_figures/codex_nuttall/gender/"
        self.path_s = self.basepath / "data/labeled_figures/codex_selden/gender/"


    def setup(self, stage):
        ## Load images into PyTorch dataset
        transform = transforms.Compose(
            [transforms.ToTensor(),
             # AddRandomBlockNoise(),
             transforms.Resize((227, 227), antialias=True),
             # transforms.Grayscale(),
             #transforms.ColorJitter(contrast=0.5),
             #transforms.RandomRotation(360),     # Maybe useful for standng and sitting
             #transforms.RandomHorizontalFlip(50),
             #transforms.RandomVerticalFlip(50)
        ])

        vindobonensis_dataset = datasets.ImageFolder(path_v, transform=transform)
        nuttall_dataset = datasets.ImageFolder(path_n, transform=transform)
        selden_dataset = datasets.ImageFolder(path_s, transform=transform)

        self.figures_dataset = ConcatDataset(
            [vindobonensis_dataset, nuttall_dataset, selden_dataset]
        )

        self.train_set, self.validation_set, self.test_set = random_split(
            figures_dataset, [0.6, 0.2, 0.2]
        )


    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False)
