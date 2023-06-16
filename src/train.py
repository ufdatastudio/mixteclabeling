import argparse
from collections import Counter
import datetime
import sys

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer

from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import BatchSizeFinder, EarlyStopping, LearningRateFinder
from pytorch_lightning import Trainer

import config
import model

from dataset import MixtecGenders
from model import NN, MixtecModel

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

class LoggingCallback(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        pl_module.logger.log_metrics(metrics, step=trainer.global_step)
        for k, v in metrics.items():
            # print(f">>>{k}: {v}")
            # print(f">>>>{dir(pl_module)}")
            # print(f">>>>>{dir(pl_module.logger)}")
            # print(f">>>>>>{dir(pl_module.logger.experiment)}")
            pl_module.logger.log_metrics({k: v}, step=trainer.global_step)
            pl_module.logger.experiment.add_scalar(k, v, trainer.global_step)
            #({k: v}, trainer.global_step)


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
    seed_everything(42, workers=True)

    # Set up logging
    logger = TensorBoardLogger(save_dir=args.logsdir, name=args.run)
    # Print log directory
    print(f"Logging to {logger.log_dir}")

    # Get the data set
    dataset = MixtecGenders(num_workers=1)

    #print(dict(Counter(dataset.targets)))

    # Configure the model
    model = NN(config.BATCH_SIZE, config.LEARNING_RATE)
    # model = MixtecModel(config.BATCH_SIZE, config.LEARNING_RATE)

    # Train the model
    early_stopping = EarlyStopping(
        monitor="val_f1",
        min_delta=1e-4,
        # stopping_threshold=1e-4,
        # divergence_threshold=9.0,
        check_finite=True,
    )

    trainer = Trainer(devices="auto", accelerator="auto", #auto_lr_find=True,
                      logger=logger, log_every_n_steps=1, 
                      min_epochs=1, max_epochs=config.EPOCHS,
                      callbacks=[
                        #   BatchSizeFinder(init_val=64),
                        # LearningRateFinder(),
                        early_stopping,
                        LoggingCallback(),
                          ])

    # Tune the model
    # trainer.tune(model, datamodule=dataset)


    # Run the evaluation

    trainer.fit(model, datamodule=dataset)
    print(trainer.validate(model, datamodule=dataset))
    print('-'*80)
    #print(trainer.predict(model, datamodule=dataset))
    #trainer.test(model, dm)

    # Run the test


if __name__ == "__main__":
    # Read commanline stuff
    main(sys.argv[1:])
