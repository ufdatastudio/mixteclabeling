from torchcam.utils import overlay_mask
from torchcam.methods import SmoothGradCAMpp

import torchvision.transforms as transforms
from torchvision.transforms.functional import normalize, to_tensor, resize, to_pil_image

from matplotlib import pyplot as plt

from PIL import Image

from typing import List

import model



def reference_class_activation_output(reference_image: Image, input_model) -> Image:
    images_from_layers_list = []

    ## TODO: This is currently hardcoded for RESNET18; need to make it dynamic for any model.
    layer_list = ['layer1', 'layer2', 'layer3', 'layer4']

    #input_model.eval()

    print(input_model.model)
    
    # Preprocess it for your chosen model
    ## TODO: Probably need to make this dynamic too
    reference_image = transform_image(reference_image)

    input_tensor = normalize(resize(reference_image, (224, 224)) / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ## TODO: Make dynamic by device
    input_tensor = input_tensor.to('cuda')
    input_model = input_model.to('cuda')

    # Preprocess your data and feed it to the model
    out = input_model(input_tensor.unsqueeze(0))

    figure = plt.figure(figsize=(10,10))

    for layer in layer_list:

        layer = 'model.' + layer

        cam_extractor = SmoothGradCAMpp(input_model, 'model.model.layer4')

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

    input_model.train()

    tensor_board_output = plot_to_image(figure)
    
    print(tensor_board_output)

    return tensor_board_output

def transform_image(image: Image) -> Image:
    to_tensor        = transforms.ToTensor()
    to_square        = transforms.Resize((224, 224), antialias=True)
    to_three_channel = transforms.Lambda(lambda x: x[:3])

    image            = to_three_channel(to_square(to_tensor(image)))
    
    return image

def plot_to_image(figure):
  """Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. The supplied figure is closed and inaccessible after this call."""
  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  # Closing the figure prevents it from being displayed directly inside
  # the notebook.
  plt.close(figure)
  buf.seek(0)
  # Convert PNG buffer to TF image
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  # Add the batch dimension
  image = tf.expand_dims(image, 0)

  return image