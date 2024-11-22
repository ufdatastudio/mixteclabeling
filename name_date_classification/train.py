# train.py

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
    def on_validation_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        for k, v in metrics.items():
            pl_module.logger.log_metrics({k: v}, step=trainer.global_step)
            
class CustomCallback(pl.Callback):
    def on_epoch_end(self, trainer, pl_module):
        # Add your custom behavior here
        print("Epoch ended!")
        # Example of accessing metrics or logging custom messages:
        metrics = trainer.callback_metrics
        print(metrics)

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
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs.")
    args = parser.parse_args(args)

    seed_everything(random.randint(0, 100), workers=True)
    logger = TensorBoardLogger(save_dir=args.logsdir, name=args.run)
    logger.log_hyperparams({"model": args.model, "batch_size": args.batch_size, "learning_rate": args.learning_rate})
    
    dataset = MixtecNameDate(batch_size=args.batch_size)
    model = m.MixtecNameDateYear(learning_rate=args.learning_rate, num_epoch=args.epochs)

    early_stopping = EarlyStopping(monitor="val_loss", patience=3, mode="min", verbose=True)
    checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints/",
    save_top_k=1,
    monitor="val_loss")
    trainer = Trainer(accelerator="auto", logger=logger, max_epochs=args.epochs, callbacks=[checkpoint_callback, early_stopping, CustomCallback(), LoggingCallback()], log_every_n_steps=10)

    trainer.callbacks.append(CustomCallback())
    print("&&&&&&", trainer.callbacks)
    train_losses, val_losses = [], []
    
    trainer.fit(model, datamodule=dataset)
    trainer = pl.Trainer(log_every_n_steps=1)
    
     # Custom hook to log losses per epoch
    # def on_epoch_end(trainer, model):
    #     train_loss = trainer.callback_metrics.get("train_loss")
    #     val_loss = trainer.callback_metrics.get("val_loss")
    #     if train_loss is not None:
    #         train_losses.append(train_loss.item())
    #     if val_loss is not None:
    #         val_losses.append(val_loss.item())
    
    print("*****", train_losses)
    
    trainer.test(model, datamodule=dataset)

    # Testing loss
    test_loss = trainer.callback_metrics.get("test_loss")
    print("Final Test Loss:", test_loss)

    # Plot the training, validation, and test losses
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(train_losses)), train_losses, label="Training Loss")
    plt.plot(range(len(val_losses)), val_losses, label="Validation Loss")
    if test_loss:
        plt.axhline(y=test_loss.item(), color="r", linestyle="--", label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training, Validation, and Test Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("Loss_Curves.png")
    plt.show()



if __name__ == "__main__":
    main(sys.argv[1:])

