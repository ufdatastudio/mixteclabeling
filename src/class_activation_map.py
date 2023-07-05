import torch

from torchcam.utils import overlay_mask
from torchcam.methods import SmoothGradCAMpp

import torchvision.transforms as transforms
from torchvision.transforms.functional import normalize, to_tensor, resize, to_pil_image
from torchvision.transforms import ToTensor

from matplotlib import pyplot as plt

from PIL import Image

import mixtec_model as m
import config

def reference_class_activation_output(reference_image: Image, input_model_path: str) -> Image:
    ## TODO: Make this dynamic by model name
    input_model = m.MixtecModel(config.LEARNING_RATE, num_epoch=config.EPOCHS, model_name='resnet18')

    input_model.load_state_dict(torch.load(input_model_path))
    
    input_model.to('cuda')    

    print(input_model.submodule_dict.keys())

    reference_image = Image.open(reference_image)

    images_from_layers_list = []

    ## TODO: This is currently hardcoded for RESNET18; need to make it dynamic for any model.
    layer_list = ['layer1', 'layer2', 'layer3', 'layer4']

    # Preprocess it for your chosen model
    ## TODO: Probably need to make this dynamic too
    reference_image = transform_image(reference_image)

    input_tensor = normalize(resize(reference_image, (224, 224)) / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ## TODO: Make dynamic by device
    # input_tensor = input_tensor.to('cuda')
    # model = model.to('cuda')

    figure = plt.figure(figsize=(60,60))

    for layer in layer_list:

        layer = 'input_model.' + layer

        cam_extractor = SmoothGradCAMpp(input_model, layer)
    
        out = input_model(input_tensor.unsqueeze(0))

        # Retrieve the CAM by passing the class index and the model output
        activation_map = cam_extractor(out.squeeze(0).argmax().item())

        plt.imshow(activation_map[0].squeeze(0).numpy()); plt.axis('off'); plt.tight_layout(); plt.show()

        # Resize the CAM and overlay it
        result = overlay_mask(to_pil_image(reference_image), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)

        images_from_layers_list.append(result)

    for i in range(4):
            # Start next subplot.
            plt.subplot(2, 2, i + 1, title=layer_list[i])
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(images_from_layers_list[i], cmap=plt.cm.binary)

    image = plot_to_Image(figure)

    return image

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