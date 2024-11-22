
# Import required packages
import matplotlib.pyplot as plt
import torchvision
import torch

# Import required modules from imported packages
from torch import nn
from torchvision import transforms
from helper_functions import set_seeds
from torch.utils.data import ConcatDataset, random_split
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from going_modular.going_modular import engine, utils
from helper_functions import plot_loss_curves
from torchinfo import summary
from datetime import datetime

# Import function to make predictions on images and plot them 
from going_modular.going_modular.predictions import pred_and_plot_image

# Config Setup
# /home/gsalunke/toblue/mixtec/mixteclabeling

train_dir = f'./sign_images/train'
test_dir = f'./sign_images/test'

# Get class names
class_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12","13","15"]

def create_dataloaders(
        train_dir: str, 
        test_dir: str, 
        transform: transforms.Compose, 
        batch_size: int, 
        num_workers: int
    ):

        # Use ImageFolder to create dataset(s)
        train_data = datasets.ImageFolder(train_dir, transform=transform)
        test_data = datasets.ImageFolder(test_dir, transform=transform)

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

        return train_dataloader, test_dataloader


if __name__ == '__main__':

    if torch.cuda.is_available():
        device = "cuda" 
        NUM_WORKERS = 16
        print(f"GPU Available for training with {NUM_WORKERS} workers")

    else:
        device= "cpu"
        print("Warning ⚠️: You are working in CPU mode. If you intended to use GPU then abort immediately by pressing ctrl + c")
        NUM_WORKERS = 6
        print(f"CPU based training with {NUM_WORKERS} workers")

    #1.🏋🏻 Get pretrained weights for ViT-Base
    print("1. Getting pretrained weights for ViT-Base")
    pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT 

    #2.🏋🏻 Setup a ViT model instance with pretrained weights
    print("2. Setting up a ViT model instance with pretrained weights ")
    pretrained_vit = torchvision.models.vit_b_16(weights=pretrained_vit_weights).to(device)

    #3. ❄️ Freeze the base parameters
    print("3. Freezing the base parameters ")
    for parameter in pretrained_vit.parameters():
        parameter.requires_grad = False
        

    #4. 🔀 Setting Random Seeds
    print("4. Setting Random Seeds")
    set_seeds()

    #5. ⚙️ Setting the in_features and out_features for vit
    print("5. Setting the in_features and out_features for vit")
    # in_feature = 16x16 x3 
    pretrained_vit.heads = nn.Linear(in_features=768, out_features=len(class_names)).to(device)


    #6. 🖨️ Print a summary using torchinfo
    print("6. Printing a summary using torchinfo")
    summary(model=pretrained_vit, 
            input_size=(32, 3, 224, 224), # (batch_size, color_channels, height, width)
            # col_names=["input_size"], # uncomment for smaller output
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"]
    )

    #7. 📂 Setup directory paths
    # Setup directory paths to train and test images
    print(f"7. Setting up train and test directories, train:{train_dir}, test:{test_dir} ")


    #8. 🏋🏻 Get automatic transforms from pretrained ViT weights
    print("8. Getting automatic transforms from pretrained ViT weights")
    pretrained_vit_transforms = pretrained_vit_weights.transforms()

    print("Pretrained VIT transforms: ")
    print(pretrained_vit_transforms)

    # 🧑🏻‍💻 9. Setup dataloaders
        # Could increase batch size if we had more samples, such as here: https://arxiv.org/abs/2205.01580 (there are other improvements there too...)
    print("9. Setting up data loaders")
    train_dataloader_pretrained, test_dataloader_pretrained = create_dataloaders(train_dir=train_dir, test_dir=test_dir, transform=pretrained_vit_transforms, batch_size=32, num_workers=NUM_WORKERS) 

    # ⨐ 10. Create optimizer and loss function
    print("10. Setting up Adam Optimiser and CrossEntropyLoss functions")
    optimizer = torch.optim.Adam(params=pretrained_vit.parameters(), 
                                lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    # 11. Train the classifier head of the pretrained ViT feature extractor model
    print("11. Beginning Model Training, This may take a while....")
    set_seeds()
    pretrained_vit_results = engine.train(model=pretrained_vit,
                                        train_dataloader=train_dataloader_pretrained,
                                        test_dataloader=test_dataloader_pretrained,
                                        optimizer=optimizer,
                                        loss_fn=loss_fn,
                                        epochs=10,
                                        device=device,
                                        class_names=class_names)   

    # 📉 12. Plotting Loss Curves
    print("12. Plotting Loss Curves")
    plot_loss_curves(pretrained_vit_results) 


    # 📁 13. Save the trained Model
    # Save the model with help from utils.py

    # Create a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Format: YYYYMMDD_HHMMSS

    # Construct the model name with the timestamp
    model_name = f"sign_classifier_vit_{timestamp}.pth"
    
    utils.save_model(model=pretrained_vit,
                    target_dir="models",
                    model_name=model_name) 
     
    print(f"Model saved to models/{model_name} ✅")