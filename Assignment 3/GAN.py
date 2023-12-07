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

results_folder = Path("./results_GAN")
results_folder.mkdir(exist_ok = True)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Training Hyperparameters
batch_size = 64   # Batch Size
z_dim = 32        # Latent Dimensionality
gen_lr = 1e-4     # Learning Rate for the Generator
disc_lr = 1e-4    # Learning Rate for the Discriminator

# Define Dataset Statistics
image_size = 32
input_channels = 1

# Resize and Normalize the Data
transform = Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Lambda(lambda t: (t * 2) - 1)
])

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

    save_image((batch['pixel_values'] + 1.) * 0.5, './results_GAN/orig.png')
    show_image((batch['pixel_values'] + 1.) * 0.5)

if __name__ == '__main__':
    visualize()

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self, z_dim, channels, generator_features=32):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(z_dim, generator_features * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(generator_features * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(generator_features * 4, generator_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(generator_features * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d( generator_features * 2, generator_features * 1, 4, 2, 1, bias=False),
            nn.BatchNorm2d(generator_features * 1),
            nn.ReLU(True),
            nn.ConvTranspose2d( generator_features * 1, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

        self.apply(weights_init)

    def forward(self, input):
        return self.model(input)

class Discriminator(nn.Module):
    def __init__(self, channels, discriminator_features=32):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(channels, discriminator_features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(discriminator_features, discriminator_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(discriminator_features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(discriminator_features * 2, discriminator_features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(discriminator_features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(discriminator_features * 4, 1, 4, 1, 0, bias=False),
        )

        self.apply(weights_init)

    def forward(self, input):
        return self.model(input)
generator = Generator(z_dim, input_channels).to(device)
discriminator = Discriminator(input_channels).to(device)

from gan_solution import *

epochs = 25

if __name__ == '__main__':
    train_dataloader, _ = get_dataloaders()
    for epoch in range(epochs):
        with tqdm(train_dataloader, unit="batch", leave=False) as tepoch:
            for batch in tepoch:
                tepoch.set_description(f"Epoch: {epoch}")

                noise = torch.randn(batch_size, z_dim, 1, 1, device=device)
                fake = generator(noise)
                real = batch['pixel_values'].to(device)

                discriminator.zero_grad()
                disc_loss = discriminator_train(discriminator, generator, real, fake.detach())
                disc_loss.backward()
                discriminator_optimizer.step()

                generator.zero_grad()
                gen_loss = generator_train(discriminator, generator, fake)
                gen_loss.backward()
                generator_optimizer.step()

                tepoch.set_postfix(disc_loss=disc_loss.item(), gen_loss=gen_loss.item())

        samples = sample(generator, 64)
        save_image((real + 1.) * 0.5, './results_GAN/orig.png')
        save_image((samples + 1.) * 0.5, f'./results_GAN/samples_{epoch}.png')

    show_image((samples + 1.) * 0.5)

from gan_solution import interpolate
if __name__ == '__main__':
    z_1 = torch.randn(1, z_dim, 1 ,1).to(device)
    z_2 = torch.randn(1, z_dim, 1, 1).to(device)

    interp = interpolate(generator, z_1, z_2, 10)
    show_image((interp + 1.) * 0.5, nrow=10)