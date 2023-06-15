import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import pytorch_lightning as pl
import torchmetrics
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score
import torchvision
from torchvision.models import get_model, get_model_weights, get_weight, list_models

import numpy as np


class MixtecModel(pl.LightningModule):
    def __init__(self, input_size, learning_rate, num_classes=2, model_name="vit_l_16"):
        super().__init__()
        self.save_hyperparameters()
        self.lr = learning_rate

        # Get models from here https://pytorch.org/vision/main/models.html
        modeloptions = ['vit_h_14', 'regnet_y_128gf', 'vit_l_16', 'regnet_y_32gf', 'regnet_y_128gf']
        # self.model = get_model(modeloptions[model_name], pretrained=True)
        self.model = get_model(model_name)

        # import inspect
        # print(inspect.getmembers(self.model))
    
    def forward(self, x):
        return self.model(x)

    # def training_step(self, batch, batch_idx):
    #     return self.training_step(batch, batch_idx)

    # def validation_step(self, batch, batch_idx):
    #     return self.model.validation_step(batch, batch_idx)

    # def test_step(self, batch, batch_idx):
    #     return self.model.test_step(batch, batch_idx)

    # def predict_step(self, batch, batch_idx):
    #     return self.model.predict_step(batch, batch_idx)    


    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,150], gamma=0.1)
        return [optimizer], [lr_scheduler]

    def _calculate_loss(self, batch, mode="train"):
        imgs, labels = batch
        preds = self.model(imgs)
        loss = F.cross_entropy(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        self.log(f'{mode}_loss', loss, sync_dist=True)
        self.log(f'{mode}_acc', acc, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")


class NN(pl.LightningModule):
    def __init__(self, input_size, learning_rate, num_classes=2):
        super().__init__()

        self.lr = learning_rate

        # self.loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1,1]))
        self.loss_fn = nn.NLLLoss()

        # Network Structure
        self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.5)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = torch.nn.Linear(16 * 56 * 56, 1568)  # Adjusted size
        self.dropout2 = nn.Dropout(0.5)
        self.final = nn.Linear(1568, 1)

        # Set up metrics
        self.metrics = MetricCollection(
            {
                "acc": Accuracy(task="binary", num_classes=num_classes),
                "rec": Recall(task="binary", num_classes=num_classes),
                "f1": F1Score(task="binary", num_classes=num_classes),
                "prec": Precision(task="binary", num_classes=num_classes),
            }
        )

    def forward(self, x):
        batch_size = x.size(0)
        print(f"The current shape is {x.shape}")
        # x = np.expand_dims (x, axis=1)
        # x = x.unsqueeze(1) # Fix for the 4D-3D problem () # https://stackoverflow.com/questions/57237381/runtimeerror-expected-4-dimensional-input-for-4-dimensional-weight-32-3-3-but
        x = self.conv1(x)
        x = F.relu(x)

        x = self.dropout1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = x.view(-1, 16 * 56 * 56)

        x = self.dropout2(x)
        x = self.fc1(x)
        x = self.final(x)
        x = x.view(batch_size, -1)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss, scores, y = self._common_step(batch, batch_idx)

        preds = torch.argmax(scores, dim=1)
        self.metrics.update(preds, y)

        self.log_dict(
            self.metrics,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            sync_dist=True
        )

        if batch_idx % 10 == 0:
            x = x[:8]
            # grid = torchvision.utils.make_grid(x.view(-1, 1, 28, 28))
            grid = torchvision.utils.make_grid(x)
            self.logger.experiment.add_image("mixtec_images", grid, self.global_step, sync_dist=True)

        return {"loss": loss, "scores": scores, "y": y}


    def _common_step(self, batch, batch_idx):
        x, y = batch
        # x = x.reshape(x.size(0), -1)
        scores = self.forward(x)
        loss = self.loss_fn(scores, y)
        return loss, scores, y

    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        # self.log("val_loss", loss)
        preds = torch.argmax(scores, dim=1)
        self.metrics.update(preds, y)
        self.log_dict(
            self.metrics,
            on_step=False,
            on_epoch=True,
            # prog_bar=True,
            sync_dist=True
        )
        return loss


    def test_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        self.log("test_loss", loss, sync_dist=True)
        return loss


    def predict_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        preds = torch.argmax(scores, dim=1)
        return preds
        

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
