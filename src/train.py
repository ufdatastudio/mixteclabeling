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
from torchcam.methods import SmoothGradCAMpp

import torchvision.transforms as transforms
from torchvision.transforms.functional import normalize, to_tensor, resize, to_pil_image
#from torchvision.models import resnet18

from matplotlib import pyplot as plt

from PIL import Image

import config
from dataset import createConfusionMatrix
from dataset import MixtecGenders
import mixtec_model as m
#from model import MixtecModel
#import class_activation_map

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
    parser.add_argument(
        "--run", default=f"mixtec-{_printdate()}", help="Name of tensorboard run."
    )
    parser.add_argument("--logsdir", default="runs/", help="Directory for logs.")
    parser.add_argument("--model", default="vgg19", help="Name of model.")
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

    trainer.fit(model, datamodule=dataset)
    
    print("Done loading")

    # Create and log confusion matrix
    logger.experiment.add_figure("Confusion matrix", createConfusionMatrix(dataset.train_dataloader(), model), config.EPOCHS)
    logger.experiment.add_figure("Confusion matrix", createConfusionMatrix(dataset.val_dataloader(), model), config.EPOCHS)
    
    # valresults = trainer.validate(model, datamodule=dataset)
    print('-'*80)
    #print(f"{valresults=}")
    print('-'*80)
    #print(trainer.predict(model, datamodule=dataset))
    #trainer.test(model, dm)

    print("Loading layer visualizations to tensorboard...")

    # model = resnet18(pretrained=True).eval()

    print(list(model.named_modules()))

    for name, module in model.named_modules():
        print(name)

    model.eval()

    reference_images = ["reference_images/female/003-a-05.png",
                        "reference_images/female/003-a-06.png",
                        "reference_images/female/072-a-04.png",
                        "reference_images/male/001-a-04.png",
                        "reference_images/male/067-a-09.png",
                        "reference_images/male/013-a-10.png",
                        ]

    # reference_images = ['reference_images/test_dog.png']

    reference_image_0 = Image.open(reference_images[0])
    reference_images_1 = Image.open(reference_images[1])
    reference_images_2 = Image.open(reference_images[2])
    reference_images_3 = Image.open(reference_images[3])

    images_from_layers_list = []

    ## TODO: This is currently hardcoded for RESNET18; need to make it dynamic for any model.
    layer_list = ['model.features.9', 'model.features.18', 'model.features.27', 'model.features.36']
    
    # Preprocess it for your chosen model
    ## TODO: Probably need to make this dynamic too
    reference_image_0 = transform_image(reference_image_0)
    reference_images_1 = transform_image(reference_images_1)
    reference_images_2 = transform_image(reference_images_2)
    reference_images_3 = transform_image(reference_images_3)

    input_tensor_0 = normalize(resize(reference_image_0, (224, 224)) / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    input_tensor_1 = normalize(resize(reference_images_1, (224, 224)) / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    input_tensor_2 = normalize(resize(reference_images_2, (224, 224)) / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    input_tensor_3 = normalize(resize(reference_images_3, (224, 224)) / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ## TODO: Make dynamic by device
    # input_tensor = input_tensor.to('cuda')
    # model = model.to('cuda')

    figure = plt.figure(figsize=(60,60))

    ### Iteration 
    layer_0 = layer_list[3]

    cam_extractor_0 = SmoothGradCAMpp(model, target_layer = layer_list)

    out_0 = model(input_tensor_0.unsqueeze(0))

    # Retrieve the CAM by passing the class index and the model output
    activation_map_0 = cam_extractor_0(out_0.squeeze(0).argmax().item(), out_0)

    # Resize the CAM and overlay it
    result_0 = overlay_mask(to_pil_image(reference_image_0), to_pil_image(activation_map_0[-1].squeeze(0), mode='F'), alpha=0.5)

    images_from_layers_list.append(result_0)

    ### Iteration 
    ### Iteration 
    layer_1 = layer_list[3]

    cam_extractor_1 = SmoothGradCAMpp(model, target_layer = layer_list)

    out_1 = model(input_tensor_1.unsqueeze(0))

    # Retrieve the CAM by passing the class index and the model output
    activation_map_1 = cam_extractor_1(out_1.squeeze(0).argmax().item(), out_1)

    # Resize the CAM and overlay it
    result_1 = overlay_mask(to_pil_image(reference_images_1), to_pil_image(activation_map_1[0].squeeze(0), mode='F'), alpha=0.5)

    images_from_layers_list.append(result_1)

    ### Iteration 
    ### Iteration 
    layer_2 = layer_list[3]

    cam_extractor_2 = SmoothGradCAMpp(model, target_layer = layer_list)

    out_2 = model(input_tensor_2.unsqueeze(0))

    # Retrieve the CAM by passing the class index and the model output
    activation_map_2 = cam_extractor_2(out_2.squeeze(0).argmax().item(), out_2)

    # Resize the CAM and overlay it
    result_2 = overlay_mask(to_pil_image(reference_images_2), to_pil_image(activation_map_2[0].squeeze(0), mode='F'), alpha=0.5)

    images_from_layers_list.append(result_2)

    ### Iteration 
    ### Iteration 
    layer_3 = "model." + layer_list[3]

    cam_extractor_3 = SmoothGradCAMpp(model = model, target_layer = layer_list)

    out_3 = model(input_tensor_3.unsqueeze(0))

    # Retrieve the CAM by passing the class index and the model output
    activation_map_3 = cam_extractor_3(out_3.squeeze(0).argmax().item(), out_3)

    #print(np.shape(activation_map_3))

    # Resize the CAM and overlay it
    result_3 = overlay_mask(to_pil_image(reference_images_3), to_pil_image(activation_map_3[0].squeeze(0), mode='F'), alpha=0.5)

    images_from_layers_list.append(result_3)

    # for layer in layer_list[:4]:

    #     #layer = getattr(model, layer)

    #     layer = "model." + layer

    #     cam_extractor = SmoothGradCAMpp(model, layer)
    
    #     out = model(input_tensor.unsqueeze(0))

    #     # Retrieve the CAM by passing the class index and the model output
    #     activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)

    #     # Resize the CAM and overlay it
    #     result = overlay_mask(to_pil_image(reference_image), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)

    #     images_from_layers_list.append(result)

    for i in range(4):
            # Start next subplot.
            plt.subplot(2, 2, i + 1, title=layer_list[i])
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(images_from_layers_list[i], cmap=plt.cm.binary)

    image = plot_to_Image(figure)

    logger.experiment.add_image("test", image, config.EPOCHS)
    
    # for reference_image_string in reference_images:

        # reference_image = Image.open(reference_image_string)

        # images_from_layers_list = []

        # ## TODO: This is currently hardcoded for RESNET18; need to make it dynamic for any model.
        # layer_list = ['features.9', 'features.18', 'features.27', 'features.36', 'features.9', 'features.18', 'features.27', 'features.36']
        
        # # Preprocess it for your chosen model
        # ## TODO: Probably need to make this dynamic too
        # reference_image = transform_image(reference_image)

        # input_tensor = normalize(resize(reference_images[0], (224, 224)) / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        # ## TODO: Make dynamic by device
        # # input_tensor = input_tensor.to('cuda')
        # # model = model.to('cuda')

        # figure = plt.figure(figsize=(60,60))

        # # for layer in layer_list[:4]:

        # #     #layer = getattr(model, layer)

        # #     layer = "model." + layer

        # #     cam_extractor = SmoothGradCAMpp(model, target_layer = layer_list)
        
        # #     out = model(input_tensor.unsqueeze(0))

        # #     # Retrieve the CAM by passing the class index and the model output
        # #     activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)

        # #     # Resize the CAM and overlay it
        # #     result = overlay_mask(to_pil_image(reference_image), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)

        # #     images_from_layers_list.append(result)

        # for i in range(4):
        #         # Start next subplot.
        #         plt.subplot(2, 2, i + 1, title=layer_list[i])
        #         plt.xticks([])
        #         plt.yticks([])
        #         plt.grid(False)
        #         plt.imshow(images_from_layers_list[i], cmap=plt.cm.binary)

        # image = plot_to_Image(figure)

        # logger.experiment.add_image(reference_image_string, image, config.EPOCHS)
            

        # flip_transform = transforms.RandomHorizontalFlip(p=1.0)

        # reference_image_flipped = flip_transform(reference_image)

        # input_tensor_flipped = normalize(resize(reference_image_flipped, (224, 224)) / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        # ## TODO: Make dynamic by device
        # # input_tensor = input_tensor.to('cuda')
        # # model = model.to('cuda')

        # for layer in layer_list[:4]:

        #     #layer = getattr(model, layer)

        #     layer = "model." + layer

        #     cam_extractor = SmoothGradCAMpp(model, layer)

        #     cam_extractor_flipped = SmoothGradCAMpp(model, layer)
        
        #     out = model(input_tensor_flipped.unsqueeze(0))

        #     # Retrieve the CAM by passing the class index and the model output
        #     activation_map_flipped = cam_extractor_flipped(out.squeeze(0).argmax().item(), out)

        #     # Resize the CAM and overlay it
        #     result_flipped = overlay_mask(to_pil_image(reference_image_flipped), to_pil_image(activation_map_flipped[0].squeeze(0), mode='F'), alpha=0.5)

        #     images_from_layers_list.append(result_flipped)

        # for i in range(8):
        #         # Start next subplot.
        #         plt.subplot(2, 4, i + 1, title=layer_list[i])
        #         plt.xticks([])
        #         plt.yticks([])
        #         plt.grid(False)
        #         plt.imshow(images_from_layers_list[i], cmap=plt.cm.binary)

        # image = plot_to_Image(figure)

        # logger.experiment.add_image(reference_image_string, image, config.EPOCHS)

    # reference_images = ["reference_images/female/003-a-06.png", 
    #                         "reference_images/female/003-a-05.png",
    #                         "reference_images/female/072-a-04.png",
    #                         "reference_images/male/001-a-04.png",
    #                         "reference_images/male/013-a-10.png",
    #                         "reference_images/male/067-a-09.png"]
    
    # savepath = "./models/mixtec_gender_classifier.pth"

    # torch.save(model.state_dict(), savepath)

    # for reference_image_string in reference_images:

    #     image = class_activation_map.reference_class_activation_output(reference_image_string, savepath)

    #     logger.experiment.add_image(image, image, config.EPOCHS)

if __name__ == "__main__":
    # Read commanline stuff
    main(sys.argv[1:])

