
# Import required packages
import matplotlib.pyplot as plt
import torchvision
import torch
import os

# Import required modules from imported packages
from torch import nn
from torchvision import transforms
from helper_functions import set_seeds
from torch.utils.data import ConcatDataset, random_split
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from going_modular.going_modular import engine
from helper_functions import plot_loss_curves
from torchinfo import summary

# Import function to make predictions on images and plot them 
from going_modular.going_modular.predictions import pred_and_plot_image


if torch.cuda.is_available():
    device = "cuda" 
    print("GPU Available for training")
    

    # 1. Get pretrained weights for ViT-Base
    pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT 

    # 2. Setup a ViT model instance with pretrained weights
    pretrained_vit = torchvision.models.vit_b_16(weights=pretrained_vit_weights).to(device)

    # 3. Freeze the base parameters
    for parameter in pretrained_vit.parameters():
        parameter.requires_grad = False
        
    # 4. Change the classifier head 
    class_names = ['name_date','year']

    set_seeds()
    pretrained_vit.heads = nn.Linear(in_features=768, out_features=len(class_names)).to(device)

    

    # Print a summary using torchinfo (uncomment for actual output)
    summary(model=pretrained_vit, 
            input_size=(32, 3, 224, 224), # (batch_size, color_channels, height, width)
            # col_names=["input_size"], # uncomment for smaller output
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"]
    )

    # Setup directory paths
    # Setup directory paths to train and test images
    train_dir = './name_date_images/train'
    test_dir = './name_date_images/test'


    # Get automatic transforms from pretrained ViT weights
    pretrained_vit_transforms = pretrained_vit_weights.transforms()
    print(pretrained_vit_transforms)

    NUM_WORKERS = 16

    def create_dataloaders(
        train_dir: str, 
        test_dir: str, 
        transform: transforms.Compose, 
        batch_size: int, 
        num_workers: int=NUM_WORKERS
    ):

        # Use ImageFolder to create dataset(s)
        train_data = datasets.ImageFolder(train_dir, transform=transform)
        test_data = datasets.ImageFolder(test_dir, transform=transform)

        # Get class names
        class_names = ['name_date', 'year']

        # Turn images into data loaders
        train_dataloader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        test_dataloader = DataLoader(
            test_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        return train_dataloader, test_dataloader, class_names


    # Setup dataloaders
    train_dataloader_pretrained, test_dataloader_pretrained, class_names = create_dataloaders(train_dir=train_dir,
                                                                                                     test_dir=test_dir,
                                                                                                     transform=pretrained_vit_transforms,
                                                                                                     batch_size=32) # Could increase if we had more samples, such as here: https://arxiv.org/abs/2205.01580 (there are other improvements there too...)
    # Create optimizer and loss function
    optimizer = torch.optim.Adam(params=pretrained_vit.parameters(), 
                                lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Train the classifier head of the pretrained ViT feature extractor model
    set_seeds()
    pretrained_vit_results = engine.train(model=pretrained_vit,
                                        train_dataloader=train_dataloader_pretrained,
                                        test_dataloader=test_dataloader_pretrained,
                                        optimizer=optimizer,
                                        loss_fn=loss_fn,
                                        epochs=10,
                                        device=device)   

    plot_loss_curves(pretrained_vit_results)   

else:
    print("Classifier cannot be trained as no GPUs were found on this system.") 

