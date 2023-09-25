import torch
import torchvision.transforms as transforms
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchvision.transforms import ToTensor
from torch import nn, optim
import pytorch_lightning as pl
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score
from torchvision.models import get_model, get_model_weights
import matplotlib.pyplot as plt 
from PIL import Image

from torchcam.utils import overlay_mask
from torchcam.methods import SmoothGradCAMpp

import io

class MixtecModel(pl.LightningModule):
    def __init__(self, learning_rate, num_classes=2, model_name="vgg16", num_epoch=1000, reference_dataloader=None):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1/382, 1/903]))
        self.num_classes = num_classes
        self.reference_dataloader = reference_dataloader

        # Get models from here https://pytorch.org/vision/main/models.html
        modeloptions = ['vit_h_14', 'regnet_y_128gf', 'vit_l_16', 'regnet_y_32gf', 'regnet_y_128gf', 'resnet18']
        
        weights = get_model_weights(model_name).DEFAULT
        self.model = get_model(model_name, weights=weights)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.hparams.learning_rate)
        
        if model_name == "vgg16":
            self.model.classifier[6] = nn.Linear(4096, 2)
            
            for i in range(0, 19):
                for param in self.model.features[i].parameters():
                    param.requires_grad = False

            # Fine tuning last layers
            for i in range(19, 30):
                for param in self.model.features[i].parameters():
                    param.requires_grad = True


        elif model_name == "vit_l_16":
            # Set the last layer
            self.model.heads = nn.Sequential()

            self.model.heads.add_module("heads", nn.Linear(1024, num_classes))

            # Fine tuning last layers
            for i, (name, param) in enumerate(self.model.named_parameters()):

                if i > 231:
                    param.requires_grad = True

                else:
                    param.requires_grad = False


        # Set up metrics
        metrics = MetricCollection(
            {
                "acc": Accuracy(task="binary", num_classes=num_classes),
                "rec": Recall(task="binary", num_classes=num_classes),
                "f1": F1Score(task="binary", num_classes=num_classes),
                "prec": Precision(task="binary", num_classes=num_classes),
            }
        )
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')

    def set_reference_dataloader(self, reference_dataloader):
        self.set_reference_dataloader = reference_dataloader
    
    def forward(self, x):
        return self.model(x)

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams, {"train_f1_epoch": 0.0,
                                                    "train_prec_epoch": 0.0,
                                                    "train_acc_epoch": 0.0,
                                                    "train_rec_epoch": 0.0,
                                                    "val_f1_epoch": 0.0,
                                                    "val_prec_epoch": 0.0,
                                                    "val_acc_epoch": 0.0,
                                                    "val_rec_epoch": 0.0
                                                    })


    def on_validation_end(self):
        print(f"self.test_metrics.compute(): {self.val_metrics.compute()}")
        print("Validation finished!")

    def _common_step(self, batch, mode="train"):
        X, y = batch
        scores = self.forward(X)
        loss = self.loss_fn(scores, y)
        return loss, scores, y
    

    def training_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, mode="train")
        preds = torch.argmax(scores, dim=1)

        self.train_metrics.update(preds, y)
        self.log_dict(
            self.train_metrics,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
            add_dataloader_idx=True
        )

        
        # if self.reference_dataloader is None:
        #     self.reference_dataloader = MixtecGenders.get_reference_dataloader()
        #     # Log the reference image
        #     for img, label in self.reference_dataloader:
        #         self.logger.experiment.add_image(f"reference_image {'female' if label.cpu() == 0 else 'male'}", img.squeeze(0), 0, dataformats="CHW")

        # if batch_idx == 0:
        #     for img, label in self.reference_dataloader:
        #         self.showActivations(img=img,
        #                 layername="conv1",
        #                 label='female' if label.cpu() == 0 else 'male',
        #                 layer=self.model.conv1,
        #                 weight=self.model.conv1.weight)
                
        return {"loss": loss, "scores": scores, "y": y}

    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, mode="val")
        preds = torch.argmax(scores, dim=1)
        self.val_metrics.update(preds, y)
        # print(f"val: {self.train_metrics.compute()}")
        self.log_dict(
            self.val_metrics,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
            add_dataloader_idx=True
        )

        return loss

    def test_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, mode="test")
        preds = torch.argmax(scores, dim=1)
        self.test_metrics.update(preds, y)
        # print(f"test: {self.test_metrics.compute()}")
        self.log_dict(
            self.test_metrics,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            add_dataloader_idx=True
        )
        return loss
    
    def configure_optimizers(self):
        return self.optimizer
    

    def reference_class_activation_output(self, reference_image: Image) -> Image:
        images_from_layers_list = []

        ## TODO: This is currently hardcoded for RESNET18; need to make it dynamic for any model.
        layer_list = ['layer1', 'layer2', 'layer3', 'layer4']

        self.model.eval()
        
        # Preprocess it for your chosen model
        ## TODO: Probably need to make this dynamic too
        reference_image = self.transform_image(reference_image)

        input_tensor = normalize(resize(reference_image, (224, 224)) / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        ## TODO: Make dynamic by device
        # input_tensor = input_tensor.to('cuda')
        # model = model.to('cuda')

        figure = plt.figure(figsize=(10,10))

        for layer in layer_list:

            layer = 'self.model.' + layer

            cam_extractor = SmoothGradCAMpp(self.model, layer)
        
            out = self.model(input_tensor.unsqueeze(0))

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

        image = self.plot_to_Image(figure)

        return image

    def transform_image(self, image: Image) -> Image:
        to_tensor        = transforms.ToTensor()
        to_square        = transforms.Resize((224, 224), antialias=True)
        to_three_channel = transforms.Lambda(lambda x: x[:3])

        image            = to_three_channel(to_square(to_tensor(image)))
        
        return image

    def plot_to_Image(self, figure) -> Image:
        """Create a pyplot plot and save to PIL ."""
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        image = Image.open(buf)
        image = ToTensor()(image)

        return image

    # def showActivations(self,
    #                     img,
    #                     label,
    #                     layername,
    #                     layer,
    #                     weight):
    #     """
    #     Usage:
    #         self.showActivations(img=self.reference_image, layername="conv1", layer=self.model.conv1, weight=self.model.conv1.weight)
    #     """

    #     # logging layer 1 activations
    #     device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #     layer1 = layer.to(device)
    #     weight1 = weight.to(device)
    #     img = img.to(device)

    #     output1 = layer1(img)

    #     output1 = output1.squeeze(0)
    #     gray_scale1 = torch.sum(output1, dim=0)
    #     gray_scale1 = gray_scale1 / output1.shape[0]

    #     self.logger.experiment.add_image(f"{label}-{layername}", gray_scale1,
    #                                      self.current_epoch, dataformats="HW")


