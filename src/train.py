import argparse
import datetime
import sys

import torch
import pytorch_lightning as pl
from pl.loggers import TensorBoardLogger
from pl.pytorch import Trainer

import config
import model


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
    parser = argparser.ArgumentParser(log_dir=f"runs/", name="mixtec-{printdate()}")
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
    logger = TensorBoardLogger(log_dir=args.logsdir, name=args.run)

    # Get the data set
    ds = MixtecGenders()

    # Configure the model
    model = NN()

    # Train the model
    trainer = Trainer(devices="auto", accelerator="auto", logger=logger)

    # Run the evaluation

    trainer.fit(model, dm)
    trainer.validate(model, dm)
    trainer.test(model, dm)

    # Run the test


if __name__ == "__main__":
    # Reach commanline stuff
    main(sys.argv[1:])
