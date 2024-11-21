import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
import pytorch_lightning as pl
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score
from torchvision.models import get_model, get_model_weights
from torch import nn, optim
from datetime import datetime
from going_modular.going_modular import utils

# Define the LightningModule for the ViT-based model
class MixtecNameDateYear(pl.LightningModule):
    def __init__(self, learning_rate=1e-3, model_name="vit_b_16", num_classes=2, num_epoch=10):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        # Load pretrained ViT weights and model
        weights = get_model_weights(model_name).DEFAULT
        self.model = get_model(model_name, weights=weights)

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Replace the classifier head
        self.model.heads = nn.Linear(in_features=768, out_features=num_classes)

        # Set up metrics
        metrics = MetricCollection({
            "acc": Accuracy(task="binary", num_classes=num_classes),
            "prec": Precision(task="binary", num_classes=num_classes),
            "rec": Recall(task="binary", num_classes=num_classes),
            "f1": F1Score(task="binary", num_classes=num_classes)
        })
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')

        # Define loss function and optimizer
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        X, y = batch
        preds = self(X)
        loss = self.loss_fn(preds, y)
        self.train_metrics.update(preds.argmax(dim=1), y)
        self.log("train_loss", loss, on_step=False, on_epoch=True) 
        
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True)
        # train_acc = self.train_metrics["train_acc"].compute()
        # train_prec = self.train_metrics["train_prec"].compute()
        # train_rec = self.train_metrics["train_rec"].compute()
        # train_f1 = self.train_metrics["train_f1"].compute()
        
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        preds = self(X)
        # print(f"preds shape: {preds.shape}, y shape: {y.shape}")

        loss = self.loss_fn(preds, y)
        self.val_metrics.update(preds.argmax(dim=1), y)
        self.log("val_loss", loss, on_step=False, on_epoch=True) 
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True)
        return self.loss_fn(preds, y)

    def test_step(self, batch, batch_idx):
        X, y = batch
        preds = self(X)
        loss = self.loss_fn(preds, y)
        self.test_metrics.update(preds.argmax(dim=1), y)
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True)
        
    # def training_epoch_end(self, outputs):
    #     # Log and print training metrics at the end of the epoch
    #     train_acc = self.train_metrics["train_acc"].compute()
    #     print(f"Epoch {self.current_epoch}: Train Accuracy: {train_acc:.4f}")
    #     self.train_metrics.reset()

    # def validation_epoch_end(self, outputs):
    #     # Log and print validation metrics at the end of the epoch
    #     val_acc = self.val_metrics["val_acc"].compute()
    #     print(f"Epoch {self.current_epoch}: Validation Accuracy: {val_acc:.4f}")
    #     self.val_metrics.reset()

    # def test_epoch_end(self, outputs):
    #     # Log and print test metrics at the end of the test epoch
    #     test_acc = self.test_metrics["test_acc"].compute()
    #     print(f"Test Accuracy: {test_acc:.4f}")
        # self.test_metrics.reset()

    def configure_optimizers(self):
        return self.optimizer


