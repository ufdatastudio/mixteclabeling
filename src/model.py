import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.transforms.functional import normalize, to_tensor, resize, to_pil_image
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import pytorch_lightning as pl
import torchmetrics
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score
import torchvision
from torchvision.models import get_model, get_model_weights, get_weight, list_models
import matplotlib.pyplot as plt 
from PIL import Image
from torchcam.utils import overlay_mask
from torchcam.methods import SmoothGradCAMpp


import numpy as np
import pandas as pd
import seaborn as sn

from dataset import MixtecGenders

class MixtecModel(pl.LightningModule):
    def __init__(self, learning_rate, num_classes=2, model_name="resnet18", num_epoch=2, reference_dataloader=None):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1/217, 1/555]))
        #self.loss_fn = nn.NLLLoss()
        self.num_classes = num_classes
        self.reference_dataloader = reference_dataloader

        # Get models from here https://pytorch.org/vision/main/models.html
        modeloptions = ['vit_h_14', 'regnet_y_128gf', 'vit_l_16', 'regnet_y_32gf', 'regnet_y_128gf', 'resnet18']
        
        weights = get_model_weights(model_name).DEFAULT
        self.model = get_model(model_name, weights=weights)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.hparams.learning_rate)


        # FIXME update the last layer for all models
        if model_name == "resnet18":
            # Set the last layer
            self.model.fc = nn.Sequential()
            self.model.fc.add_module("maxpool", nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False))
            self.model.fc.add_module("conv1", nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False))
            self.model.fc.add_module("fc", nn.Linear(512, num_classes))
            
            # Fine tuning the last layer
            plist = [{'params': self.model.fc.parameters(), 'lr': 1e-2} ]
            self.optimizer = optim.AdamW(plist, lr=self.hparams.learning_rate)

        # Set up metrics
        metrics = MetricCollection(
            {
                "acc": Accuracy(task="binary", num_classes=num_classes),
                "rec": Recall(task="binary", num_classes=num_classes),
                "f1": F1Score(task="binary", num_classes=num_classes),
                "prec": Precision(task="binary", num_classes=num_classes),
            }
        )
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')

    def set_reference_dataloader(self, reference_dataloader):
        self.set_reference_dataloader = reference_dataloader
    
    def forward(self, x):
        return self.model(x)

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams, {"train_f1_epoch": 0.0,
                                                    "train_prec_epoch": 0.0,
                                                    "train_acc_epoch": 0.0,
                                                    "train_rec_epoch": 0.0,
                                                    "val_f1_epoch": 0.0,
                                                    "val_prec_epoch": 0.0,
                                                    "val_acc_epoch": 0.0,
                                                    "val_rec_epoch": 0.0
                                                    })


    def on_validation_end(self):
        print(f"self.test_metrics.compute(): {self.val_metrics.compute()}")
        print("Validation finished!") 

    def _common_step(self, batch, mode="train"):
        X, y = batch
        scores = self.forward(X)
        loss = self.loss_fn(scores, y)
        return loss, scores, y
    

    def training_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, mode="train")
        preds = torch.argmax(scores, dim=1)

        self.train_metrics.update(preds, y)
        self.log_dict(
            self.train_metrics,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
            add_dataloader_idx=True
        )

        # if self.reference_dataloader is None:
        #     self.reference_dataloader = MixtecGenders.get_reference_dataloader()
        #     # Log the reference image
        #     for img, label in self.reference_dataloader:
        #         self.logger.experiment.add_image(f"reference_image {'female' if label.cpu() == 0 else 'male'}", img.squeeze(0), 0, dataformats="CHW")

        # if batch_idx == 0:
        #     for img, label in self.reference_dataloader:
        #         self.showActivations(img=img,
        #                 layername="conv1",
        #                 label='female' if label.cpu() == 0 else 'male',
        #                 layer=self.model.conv1,
        #                 weight=self.model.conv1.weight)
                
        return {"loss": loss, "scores": scores, "y": y}

    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, mode="val")
        preds = torch.argmax(scores, dim=1)
        self.val_metrics.update(preds, y)
        # print(f"val: {self.train_metrics.compute()}")
        self.log_dict(
            self.val_metrics,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
            add_dataloader_idx=True
        )
        return loss

    def test_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, mode="test")
        preds = torch.argmax(scores, dim=1)
        self.test_metrics.update(preds, y)
        # print(f"test: {self.test_metrics.compute()}")
        self.log_dict(
            self.test_metrics,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            add_dataloader_idx=True
        )
        return loss
    
    def configure_optimizers(self):
        return self.optimizer


    def showActivations(self,
                        img,
                        label,
                        layername,
                        layer,
                        weight):
        """
        Usage:
            self.showActivations(img=self.reference_image, layername="conv1", layer=self.model.conv1, weight=self.model.conv1.weight)
        """

        # logging layer 1 activations
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        layer1 = layer.to(device)
        weight1 = weight.to(device)
        img = img.to(device)

        output1 = layer1(img)

        output1 = output1.squeeze(0)
        gray_scale1 = torch.sum(output1, dim=0)
        gray_scale1 = gray_scale1 / output1.shape[0]

        self.logger.experiment.add_image(f"{label}-{layername}", gray_scale1,
                                         self.current_epoch, dataformats="HW")
        
