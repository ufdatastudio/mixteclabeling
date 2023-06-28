import argparse
from collections import Counter
import datetime
import os
import sys
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torchvision.transforms.functional import normalize, to_tensor, resize, to_pil_image

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import BatchSizeFinder, EarlyStopping, LearningRateFinder
from pytorch_lightning import Trainer

from torchcam.utils import overlay_mask
from torchcam.methods import SmoothGradCAMpp
from grad_cam import grad_cam

import torchvision.transforms as transforms
from torchvision.transforms.functional import normalize, to_tensor, resize, to_pil_image
from torchvision.models import resnet18

from matplotlib import pyplot as plt

from PIL import Image

import config
from dataset import createConfusionMatrix
from dataset import MixtecGenders
import model as m
#from model import MixtecModel
import class_activation_map

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
            pl_module.logger.log_metrics({k: v}, step=trainer.global_step)
            pl_module.logger.experiment.add_scalar(k, v, trainer.global_step)


def main(args):
    # Config stuff --------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_false")
    parser.add_argument("--validate", action="store_false")
    parser.add_argument("--test", action="store_false")
    parser.add_argument(
        "--run", default=f"mixtec-{_printdate()}", help="Name of tensorboard run."
    )
    parser.add_argument("--logsdir", default="runs/", help="Directory for logs.")
    parser.add_argument("--model", default="resnet18", help="Name of model.")
    args = parser.parse_args(args)

    # Deep Learning stuff ---------------
    seed_everything(42, workers=True)
    torch.set_float32_matmul_precision('medium')

    # Set up logging
    logger = TensorBoardLogger(save_dir=args.logsdir, name=args.run, default_hp_metric=False)
    # Print log directory
    print(f">>> # Logging to {logger.log_dir}")
    logger.log_hyperparams({"model": args.model})

    num_workers = len(os.sched_getaffinity(0))
    print(f">>> # Using {num_workers} workers")
    # Get the data set
    # Using only one worker is faster
    dataset = MixtecGenders(num_workers=1, batch_size=config.BATCH_SIZE)

    #print(dict(Counter(dataset.targets)))

    # Configure the model
    #model = NN(config.BATCH_SIZE, config.LEARNING_RATE)
    model = m.MixtecModel(config.LEARNING_RATE, num_epoch=config.EPOCHS, model_name=args.model)
    #model = resnet18(pretrained=True).eval()
    # model.set_reference_dataloader(dataset.reference_dataloader)

    # Train the model
    early_stopping = EarlyStopping(
        monitor="val_f1",
        # min_delta=1e-6,
        stopping_threshold=1e-4,
        # divergence_threshold=9.0,
        check_finite=True,
    )

    trainer = Trainer(devices="auto", accelerator="auto", #auto_lr_find=True,
                      logger=logger, log_every_n_steps=1, enable_progress_bar=True,
                      min_epochs=1, max_epochs=config.EPOCHS,
                      callbacks=[
                        #   BatchSizeFinder(init_val=64),
                        # LearningRateFinder(),
                        #early_stopping,
                        # LoggingCallback(),
                          ])

    # Tune the model
    # trainer.tune(model, datamodule=dataset)


    # Run the evaluation

    fitresults = trainer.fit(model, datamodule=dataset)

    print("Loading layer visualizations to tensorboard...")
    reference_image = Image.open("reference_images/female/003-a-06.png")
    # layer_output   = class_activation_map.reference_class_activation_output(reference_image, model)
    # logger.experiment.add_image("layer output", layer_output, model.current_epoch, dataformats="HW")


    # images_from_layers_list = []

    # ## TODO: This is currently hardcoded for RESNET18; need to make it dynamic for any model.
    # layer_list = ['layer1', 'layer2', 'layer3', 'layer4']

    # model.eval()
    
    # # Preprocess it for your chosen model
    # ## TODO: Probably need to make this dynamic too
    # reference_image = transform_image(reference_image)

    # input_tensor = normalize(resize(reference_image, (224, 224)) / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    # ## TODO: Make dynamic by device
    # input_tensor = input_tensor.to('cuda')
    # model = model.to('cuda')

    # # Preprocess your data and feed it to the model
    # out = model(input_tensor.unsqueeze(0))

    # figure = plt.figure(figsize=(10,10))

    # for layer in layer_list:

    #     layer = 'model.' + layer

    #     cam_extractor = SmoothGradCAMpp(model, 'model.layer4')
    
    #     out = model(input_tensor.unsqueeze(0))

    #     # Retrieve the CAM by passing the class index and the model output
    #     activation_map = cam_extractor(out.squeeze(0).argmax().item())

    #     plt.imshow(activation_map[0].squeeze(0).numpy()); plt.axis('off'); plt.tight_layout(); plt.show()

    #     # Resize the CAM and overlay it
    #     result = overlay_mask(to_pil_image(reference_image), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)

    #     images_from_layers_list.append(result)

    # for i in range(4):
    #         # Start next subplot.
    #         plt.subplot(2, 2, i + 1, title=layer_list[i])
    #         plt.xticks([])
    #         plt.yticks([])
    #         plt.grid(False)
    #         plt.imshow(images_from_layers_list[i], cmap=plt.cm.binary)

    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.RandomErasing(),
        # transforms.Resize((224, 224), antialias=True),
        # transforms.Lambda(lambda x: x[:3])
        # # transforms.CenterCrop(224),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    input_tensor = transform(reference_image)

    cam = grad_cam(model.model, input_tensor, model.model.layer4, "Mixtec_Woman")

    print(cam)

    # model.train()

    #tensor_board_output = plot_to_image(figure)


    
    print("Done loading")

    # Create and log confusion matrix
    logger.experiment.add_figure("Confusion matrix", createConfusionMatrix(dataset.train_dataloader(), model), config.EPOCHS)
    logger.experiment.add_figure("Confusion matrix", createConfusionMatrix(dataset.val_dataloader(), model), config.EPOCHS)
    
    valresults = trainer.validate(model, datamodule=dataset)
    print('-'*80)
    print(f"{valresults=}")
    print('-'*80)
    #print(trainer.predict(model, datamodule=dataset))
    #trainer.test(model, dm)

if __name__ == "__main__":
    # Read commanline stuff
    main(sys.argv[1:])

