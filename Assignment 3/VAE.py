import random
import numpy as np
from tqdm.auto import tqdm
from datasets import load_dataset

import torch

from torch import nn
import torch.nn.functional as F
from torch.optim import Adam

from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt

from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
from torchvision import transforms

from pathlib import Path
from torch.utils.data import DataLoader

def fix_experiment_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

fix_experiment_seed()

results_folder = Path("./results_VAE")
results_folder.mkdir(exist_ok = True)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Training Hyperparameters
batch_size = 64   # Batch Size
z_dim = 32        # Latent Dimensionality
lr = 1e-4         # Learning Rate

# Define Dataset Statistics
image_size = 32
input_channels = 1

# Resize and Normalize the Data
transform = Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Lambda(lambda t: (t * 2) - 1)]
)

# Helper Functions
def show_image(image, nrow=8):
    # Input: image
    # Displays the image using matplotlib
    grid_img = make_grid(image.detach().cpu(), nrow=nrow, padding=0)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.axis('off')

def transforms_examples(examples):
    # Helper function to perform transformations on the input images
    if "image" in examples:
        examples["pixel_values"] = [transform(image) for image in examples["image"]]
        del examples["image"]
    else:
        examples["pixel_values"] = [transform(image) for image in examples["img"]]
        del examples["img"]

    return examples

# Load dataset from the hub, normalize it and create the dataloader
def get_dataloaders():
    dataset = load_dataset("mnist", cache_dir='./data')
    transformed_dataset = dataset.with_transform(transforms_examples)
    train_dataloader = DataLoader(transformed_dataset["train"], batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(transformed_dataset["test"], batch_size=batch_size, shuffle=False, drop_last=False)

    return train_dataloader, test_dataloader

# Visualize the Dataset
def visualize():
    train_dataloader, _ = get_dataloaders()
    batch = next(iter(train_dataloader))
    print(batch['pixel_values'].shape)

    save_image((batch['pixel_values'] + 1.) * 0.5, './results_VAE/orig.png')
    show_image((batch['pixel_values'] + 1.) * 0.5)

if __name__ == '__main__':
    visualize()

from vae_solution import Encoder, Decoder
from vae_solution import DiagonalGaussianDistribution
from vae_solution import VAE

if __name__ == '__main__':
    model = VAE(in_channels=input_channels,
                input_size=image_size,
                z_dim=z_dim,
                decoder_features=32,
                encoder_features=32,
                device=device
                )
    model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)

epochs = 25

if __name__ == '__main__':
    train_dataloader, _ = get_dataloaders()
    for epoch in range(epochs):
        with tqdm(train_dataloader, unit="batch", leave=False) as tepoch:
            model.train()
            for batch in tepoch:
                tepoch.set_description(f"Epoch: {epoch}")

                optimizer.zero_grad()

                batch_size = batch["pixel_values"].shape[0]
                x = batch["pixel_values"].to(device)

                recon, nll, kl = model(x)
                loss = (nll + kl).mean()

                loss.backward()
                optimizer.step()

                tepoch.set_postfix(loss=loss.item())

        samples = model.sample(batch_size=64)
        save_image((x + 1.) * 0.5, f'./results_VAE/orig_{epoch}.png')
        save_image((recon + 1.) * 0.5, f'./results_VAE/recon_{epoch}.png')
        save_image((samples + 1.) * 0.5, f'./results_VAE/samples_{epoch}.png')

    show_image(((samples + 1.) * 0.5).clamp(0., 1.))
from vae_solution import interpolate

from vae_solution import interpolate

if name == 'main':
    z_1 = torch.randn(1, z_dim).to(device)
    z_2 = torch.randn(1, z_dim).to(device)

    interp = interpolate(model, z_1, z_2, 10)
    # show_image((interp + 1.) * 0.5, nrow=10)
    save_image((interp + 1.) * 0.5, f'./results_VAE/interpolate.png')