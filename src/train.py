import argparse
import datetime
import sys

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer

import config
import model

from dataset import MixtecGenders
from model import NN

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


def main(args):
    # Config stuff --------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_false")
    parser.add_argument("--validate", action="store_false")
    parser.add_argument("--test", action="store_false")
    parser.add_argument(
        "--run", default=f"mixtec-{_printdate()}", help="Name of tensorboard run"
    )
    parser.add_argument("--logsdir", default="runs/", help="Directory for logs")
    args = parser.parse_args(args)

    # Deep Learning stuff ---------------
    # Set up logging
    logger = TensorBoardLogger(save_dir=args.logsdir, name=args.run)

    # Get the data set
    dataset = MixtecGenders()

    # Configure the model
    model = NN(config.BATCH_SIZE, config.LEARNING_RATE)

    # Train the model
    trainer = Trainer(devices="auto", accelerator="auto", gpus="auto", 
                      logger=logger, log_every_n_steps=1, 
                      min_epochs=1, max_epochs=config.EPOCHS)

    # Run the evaluation

    trainer.fit(model, dataset)
    trainer.validate(model, dataset)
    #trainer.test(model, dm)

    # Run the test


if __name__ == "__main__":
    # Read commanline stuff
    main(sys.argv[1:])
