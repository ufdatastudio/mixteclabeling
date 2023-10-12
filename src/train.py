import argparse
from collections import Counter
import datetime
import os
import io
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torchvision.transforms.functional import normalize, to_tensor, resize, to_pil_image
from torchvision.transforms import ToTensor

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import BatchSizeFinder, EarlyStopping, LearningRateFinder
from pytorch_lightning import Trainer

from torchcam.utils import overlay_mask
from torchcam.methods import GradCAM

import torchvision.transforms as transforms
from torchvision.transforms.functional import normalize, to_tensor, resize, to_pil_image

from matplotlib import pyplot as plt

from PIL import Image

import config
from dataset import createConfusionMatrix
from dataset import MixtecGenders
import mixtec_model as m

def _printdate(dt=datetime.datetime.now()):
    """print a date and time string containing only numbers and dashes"""

    # your code here
    if dt.hour < 10:
        hour = "0" + str(dt.hour)
    else:
        hour = str(dt.hour)

    if dt.minute < 10:
        minute = "0" + str(dt.minute)
    else:
        minute = str(dt.minute)

    d = "{}-{}-{}-{}-{}".format(str(dt.month), str(dt.day), str(dt.year), hour, minute)
    return d

def transform_image(image: Image) -> Image:
    to_tensor        = transforms.ToTensor()
    to_square        = transforms.Resize((224, 224), antialias=True)
    to_three_channel = transforms.Lambda(lambda x: x[:3])

    image            = to_three_channel(to_square(to_tensor(image)))
    
    return image

def plot_to_Image(figure) -> Image:
    """Create a pyplot plot and save to PIL ."""
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    image = Image.open(buf)
    image = ToTensor()(image)

    return image



class LoggingCallback(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        pl_module.logger.log_metrics(metrics, step=trainer.global_step)
        for k, v in metrics.items():
            pl_module.logger.log_metrics({k: v}, step=trainer.global_step)
            pl_module.logger.experiment.add_scalar(k, v, trainer.global_step)


def main(args):
    # Config stuff --------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_false")
    parser.add_argument("--validate", action="store_false")
    parser.add_argument("--test", action="store_false")
    parser.add_argument("--run", default=f"mixtec-{_printdate()}", help="Name of tensorboard run.")
    parser.add_argument("--logsdir", default="test/", help="Directory for logs.")
    parser.add_argument("--model", default="vgg16", help="Name of model.")
    parser.add_argument("--batch_size", default=64, help="Batch size.")
    parser.add_argument("--learning_rate", default=0.01, help="Learning rate.")
    parser.add_argument("--epochs", default=100, help="Number of epochs.")
    parser.add_argument("--transforms", default="", help="Transforms to apply.")
    parser.add_argument("--category", default="pose", help="Category to train on.")
    args = parser.parse_args(args)


    args.logsdir = args.logsdir if args.logsdir else config.RUNS_DIR
    args.learning_rate = args.learning_rate if args.learning_rate else config.LEARNING_RATE
    args.batch_size = args.batch_size if args.batch_size else config.BATCH_SIZE
    args.epochs = args.epochs if args.epochs else config.EPOCHS
    args.model = args.model if args.model else config.MODEL
    args.transforms = args.transforms if args.transforms else config.TRANSFORMS
    args.category = args.category if args.category else config.CATEGORY

    # Deep Learning stuff ---------------
    seed_everything(42, workers=True)
    torch.set_float32_matmul_precision('medium')

    # Set up logging
    logger = TensorBoardLogger(save_dir=args.logsdir, name=args.run, default_hp_metric=False)
    # Print log directory
    print(f">>> # Logging to {logger.log_dir}")
    logger.log_hyperparams({"model": args.model})
    logger.log_hyperparams({"batch_size": args.batch_size})
    logger.log_hyperparams({"learning_rate": args.learning_rate})

    num_workers = len(os.sched_getaffinity(0))
    print(f">>> # Using {num_workers} workers")

    dataset = MixtecGenders(num_workers=1, batch_size=int(args.batch_size), input_transforms=args.transforms.split("_"), category=args.category)

    model = m.MixtecModel(learning_rate=float(args.learning_rate), num_epoch=int(args.epochs), model_name=args.model)


    # Train the model
    early_stopping = EarlyStopping(
        monitor="val_f1",
        # min_delta=1e-6,
        stopping_threshold=1e-6,
        # divergence_threshold=9.0,
        check_finite=True,
    )

    trainer = Trainer(devices="auto", accelerator="auto", #auto_lr_find=True,
                      logger=logger, log_every_n_steps=1, enable_progress_bar=True,
                      min_epochs=1, max_epochs=args.epochs,
                      callbacks=[
                          ])


    trainer.fit(model, datamodule=dataset)
    
    print("Done loading")

    # Create and log confusion matrix
    logger.experiment.add_figure("Confusion matrix", createConfusionMatrix(dataset.train_dataloader(), model), args.epochs)
    logger.experiment.add_figure("Confusion matrix", createConfusionMatrix(dataset.val_dataloader(), model), args.epochs)
    
    # valresults = trainer.validate(model, datamodule=dataset)
    print('-'*80)
    #print(f"{valresults=}")
    print('-'*80)
    #print(trainer.predict(model, datamodule=dataset))
    trainer.test(model, datamodule=dataset)

    logger.experiment.add_figure("Confusion matrix", createConfusionMatrix(dataset.val_dataloader(), model), args.epochs)

if __name__ == "__main__":
    # Read commanline stuff
    main(sys.argv[1:])