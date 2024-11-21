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
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split
import torchvision.transforms as transforms
import pytorch_lightning as pl
from datasets import load_dataset

class CustomImageDataset(Dataset):
    def __init__(self, dataset, transform=None):
        """
        Custom dataset to load images from Hugging Face dataset and apply transformations.
        Args:
            dataset: Hugging Face dataset containing file paths in 'file_name' column.
            transform: Transformation to apply on each image.
        """
        self.dataset = dataset
        self.transform = transform
        self.label_map = {'name-date': 0, 'year': 1}
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image = self.dataset[idx]['image']  # Assuming 'image' field contains image data
        has_A0 = self.dataset[idx]['has_AO']
        if has_A0 == 0:
            label = 0
        else:
            label = 1
        
        # If the image data is in binary format, convert it to a PIL image
        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image)).convert("RGB")
        
        # Apply transformations if any
        if self.transform:
            image = self.transform(image)
        
        return image, label

class MixtecNameDate(pl.LightningDataModule):
    def __init__(self, data_dir=None, batch_size=125, num_workers=8, input_transforms=None):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        # self.category = str(category)
        
        self.reference_dataloader = None

        # Set default transforms
        self.input_transforms = transforms.Compose(
            [
                transforms.Lambda(lambda x: x.convert("RGB") if x.mode != "RGB" else x),
                transforms.ToTensor(),
                transforms.Resize((224, 224), antialias=True),
            ]
        )

    def prepare_data(self):
        # Load Hugging Face dataset
        self.hf_dataset = load_dataset("ufdatastudio/mixtec-zouche-nuttall-british-museum", data_dir="name-date-cutouts", revision="main", split="train")

    def setup(self, stage):
        # Initialize datasets with transformations
        transform = self.input_transforms

        # Create custom dataset from Hugging Face data
        custom_dataset = CustomImageDataset(self.hf_dataset, transform=transform)

        # Split the dataset into training, validation, and test sets
        train_size = int(0.6 * len(custom_dataset))
        val_size = int(0.2 * len(custom_dataset))
        test_size = len(custom_dataset) - train_size - val_size
        self.train_set, self.val_set, self.test_set = random_split(
            custom_dataset, [train_size, val_size, test_size]
        )

        if stage == 'fit' or stage is None:
            train_labels = [item[1] for item in self.train_set]
            print(f"Training Set: {Counter(train_labels)}")

            val_labels = [item[1] for item in self.val_set]
            print(f"Validation Set: {Counter(val_labels)}")

        if stage == 'test' or stage is None:
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
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

def createConfusionMatrix(loader, net):
    y_pred = []  # Save predictions
    y_true = []  # Save ground truth

    for inputs, labels in loader:
        output = net(inputs)  # Feed Network
        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output)  # Save prediction
        labels = labels.data.cpu().numpy()
        y_true.extend(labels)  # Save ground truth

    classes = ("name-date", "year")  # Define class labels
    cf_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix, index=[i for i in classes],
                         columns=[i for i in classes])
    plt.figure(figsize=(12, 7))    
    return sn.heatmap(df_cm, annot=True).get_figure()
    
def main():
    # Set parameters for the data module
    batch_size = 125
    num_workers = 8
    input_transforms = ['RandomHorizontalFlip', 'RandomVerticalFlip']  # Example transforms

    # Initialize data module
    data_module = MixtecNameDate(batch_size=batch_size, num_workers=num_workers)

    # Prepare and setup data
    data_module.prepare_data()
    data_module.setup(stage='test')

    # Get train dataloader and print a batch to verify
    train_loader = data_module.train_dataloader()
    for batch in train_loader:
        images, labels = batch
        print("Batch of images:", images.shape)
        print("Batch of labels:", labels)
        break  # Print only the first batch for verification

if __name__ == "__main__":
    main()

