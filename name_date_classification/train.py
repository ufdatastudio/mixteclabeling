import argparse
import datetime
import os
import io
import sys
import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image
import torch
from torchvision.transforms import ToTensor
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from dataset import createConfusionMatrix, MixtecNameDate
import mixtec_name_date_year as m
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything

def _printdate(dt=datetime.datetime.now()):
    hour = f"{dt.hour:02}"
    minute = f"{dt.minute:02}"
    return f"{dt.month}-{dt.day}-{dt.year}-{hour}-{minute}"
    

class LoggingCallback(pl.Callback):
    def __init__(self):
        self.collection = []
   
    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        train_loss = trainer.callback_metrics.get("train_loss")
        train_acc = trainer.callback_metrics.get("train_acc")
        print(f"Training metrics after epoch end train_loss : {train_loss} train_acc : {train_acc}")
        
    def on_validation_end(self, trainer, pl_module):
        elogs = trainer.logged_metrics # access it here
        self.collection.append(elogs)
        metrics = trainer.callback_metrics
        val_loss = trainer.callback_metrics.get("val_loss")
        val_acc = trainer.callback_metrics.get("val_acc")
        print(f"Validation metrics after epoch end val_loss : {val_loss} val_acc : {val_acc}")
        for k, v in metrics.items():
            pl_module.logger.log_metrics({k: v}, step=trainer.global_step)
            
    def on_test_epoch_end(self, trainer, pl_module):
        pass

def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--run", default=f"mixtec-{_printdate()}", help="Name of tensorboard run.")
    parser.add_argument("--logsdir", default="out/transform_test/", help="Directory for logs.")
    parser.add_argument("--model", default="vgg16", help="Name of model.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs.")
    args = parser.parse_args(args)

    seed_everything(random.randint(0, 100), workers=True)
    logger = TensorBoardLogger(save_dir=args.logsdir, name=args.run)
    logger.log_hyperparams({"model": args.model, "batch_size": args.batch_size, "learning_rate": args.learning_rate})
    
    dataset = MixtecNameDate(batch_size=args.batch_size)
    model = m.MixtecNameDateYear(learning_rate=args.learning_rate, num_epoch=args.epochs)

    early_stopping = EarlyStopping(monitor="val_loss", patience=3, mode="min", verbose=True)
    cb = LoggingCallback()
 
    trainer = Trainer(accelerator="auto", logger=pl.loggers.TensorBoardLogger('tb_logs/'), max_epochs=args.epochs, callbacks=[early_stopping, cb], log_every_n_steps=10)
    
    trainer.fit(model, datamodule=dataset)
    trainer.test(model, datamodule=dataset)
    

    # Testing loss
    test_loss = trainer.callback_metrics.get("test_loss")
    test_acc = trainer.callback_metrics.get("test_acc")
    print("Final Test Loss:", test_loss)
    print("Final Test Accuracy:", test_acc)
    
    # Print plots
    metrics = cb.collection
    epochs = range(1, len(metrics))
    val_loss = []
    for entry in metrics:
        val_loss.append(entry['val_loss'].item())
    val_acc = [entry['val_acc'].item() for entry in metrics]
    train_loss = [entry.get('train_loss').item() for entry in metrics if 'train_loss' in entry]
    train_acc = [entry.get('train_acc').item() for entry in metrics if 'train_acc' in entry]


if __name__ == "__main__":
    main(sys.argv[1:])

