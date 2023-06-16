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
import matplotlib.pyplot as plt 

import numpy as np


class MixtecModel(pl.LightningModule):
    def __init__(self, input_size, learning_rate, num_classes=2, model_name="vit_l_16"):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.loss_fn = nn.CrossEntropyLoss()
        #self.loss_fn = nn.NLLLoss()

        self.reference_image = None
        self.reference_label = None

        # Get models from here https://pytorch.org/vision/main/models.html
        modeloptions = ['vit_h_14', 'regnet_y_128gf', 'vit_l_16', 'regnet_y_32gf', 'regnet_y_128gf']
        # self.model = get_model(modeloptions[model_name], pretrained=True)
        self.model = get_model(model_name)

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
    
    def forward(self, x):
        return self.model(x)

    def _common_step(self, batch, mode="train"):
        X, y = batch
        scores = self.forward(X)
        loss = self.loss_fn(scores, y)
        return loss, scores, y

    def training_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, mode="train")
        preds = torch.argmax(scores, dim=1)

        self.train_metrics.update(preds, y)
        # print(f"train: {self.train_metrics.compute()}")
        self.log_dict(
            self.train_metrics,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
            add_dataloader_idx=True
        )
        return {"loss": loss, "scores": scores, "y": y}

    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, mode="val")
        preds = torch.argmax(scores, dim=1)
        self.val_metrics.update(preds, y)
        # print(f"val: {self.val_metrics.compute()}")
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
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        # lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,150], gamma=0.1)
        # return [optimizer], [lr_scheduler]
        return optimizer
        


class NN(pl.LightningModule):
    def __init__(self, input_size, learning_rate, num_classes=2):
        super().__init__()
        self.save_hyperparameters()

        # self.lr = learning_rate

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


    def makegrid(self, output,numrows):
        outer=(torch.Tensor.cpu(output).detach())
        plt.figure(figsize=(20,5))
        b=np.array([]).reshape(0,outer.shape[2])
        c=np.array([]).reshape(numrows*outer.shape[2],0)
        i=0
        j=0
        while(i < outer.shape[1]):
            img=outer[0][i]
            b=np.concatenate((img,b),axis=0)
            j+=1
            if(j==numrows):
                c=np.concatenate((c,b),axis=1)
                b=np.array([]).reshape(0,outer.shape[2])
                j=0
                 
            i+=1
        return c

    def showActivations(self,x):
            # logging reference image
            #print(x.shape)
            #print(torch.Tensor.cpu(x[0][0]).shape)
            #x = transforms.Grayscale()(x)

            #print(x.shape)
            #self.logger.experiment.add_image("input",torch.Tensor.cpu(x[0][0]),self.current_epoch,dataformats="HW")
 
            print("<<<<<< reached showActivations >>>>>>")

            # logging layer 1 activations        
            out = self.fc1(x)
            c=self.makegrid(out,4)
            self.logger.experiment.add_image("layer 1",c,self.current_epoch,dataformats="HW")
             
            # logging layer 2 activations        
            # out = self.conv2(out)
            # c=self.makegrid(out,8)
            # self.logger.experiment.add_image("layer 2",c,self.current_epoch,dataformats="HW")

    def forward(self, x):
        batch_size = x.size(0)
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

        if batch_idx == 0:
            self.reference_image = x[0]
            self.reference_label = y[0]

        preds = torch.argmax(scores, dim=1)
        self.train_metrics.update(preds, y)

        self.log_dict(
            self.train_metrics,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True
        )

        if batch_idx % 10 == 0:
            x = x[:8]
            # grid = torchvision.utils.make_grid(x.view(-1, 1, 28, 28))
            grid = torchvision.utils.make_grid(x)
            self.logger.experiment.add_image("mixtec_images", grid, self.global_step)
            print("<<<<< reached on_training_epoch_end >>>>>")
            self.showActivations(self.reference_image)

        return {"loss": loss, "scores": scores, "y": y}

    def on_training_epoch_end(self, outputs):
        self.train_metrics.reset()
    
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
        self.val_metrics.update(preds, y)
        self.log_dict(
            self.val_metrics,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True
        )
        return loss
    
        
    # def validation_epoch_end(self, outputs):
    #     self.log('valid_acc_epoch', self.val_metrics.compute())
    #     self.val_metrics.reset()



    def test_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        preds = torch.argmax(scores, dim=1)
        self.test_metrics.update(preds, y)
        self.log_dict(
            self.test_metrics,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True
        )
        return loss


    def predict_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        preds = torch.argmax(scores, dim=1)
        return preds
        

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)