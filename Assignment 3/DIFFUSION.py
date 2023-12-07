import random
import numpy as np
from tqdm.auto import tqdm
from datasets import load_dataset
from inspect import isfunction
from functools import partial
import math
from einops import rearrange

import torch
from torch import nn, einsum
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

results_folder = Path("./results_diffusion")
results_folder.mkdir(exist_ok = True)
device = "cuda" if torch.cuda.is_available() else "cpu"


# Training Hyperparameters
batch_size = 64   # Batch Size
z_dim = 32        # Latent Dimensionality
lr = 1e-4         # Learning Rate

# Hyperparameters taken from Ho et. al for noise scheduling
T = 1000            # Diffusion Timesteps
beta_start = 0.0001 # Starting variance
beta_end = 0.02     # Ending variance

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

    save_image((batch['pixel_values'] + 1.) * 0.5, './results_diffusion/orig.png')
    show_image((batch['pixel_values'] + 1.) * 0.5)

if __name__ == '__main__':
    visualize()

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim):
    return nn.ConvTranspose2d(dim, dim, 4, 2, 1)

def Downsample(dim):
    return nn.Conv2d(dim, dim, 4, 2, 1)

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    """https://arxiv.org/abs/1512.03385"""

    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out))
            if exists(time_emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.block1(x)

        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            h = rearrange(time_emb, "b c -> b c 1 1") + h

        h = self.block2(h)
        return h + self.res_conv(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1),
                                    nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class Unet(nn.Module):
    def __init__(
            self,
            dim,
            init_dim=None,
            out_dim=None,
            dim_mults=(1, 2, 4, 8),
            channels=3,
            with_time_emb=True,
            resnet_block_groups=8,
    ):
        super().__init__()

        # determine dimensions
        self.channels = channels

        init_dim = default(init_dim, dim // 3 * 2)
        self.init_conv = nn.Conv2d(channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings
        if with_time_emb:
            time_dim = dim * 4
            self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(dim),
                nn.Linear(dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )
        else:
            time_dim = None
            self.time_mlp = None

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out * 2, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Upsample(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            block_klass(dim, dim), nn.Conv2d(dim, out_dim, 1)
        )

    def forward(self, x, time):
        # Returns the noise prediction from the noisy image x at time t
        # Inputs:
        #   x: noisy image tensor of size (batch_size, 3, 32, 32)
        #   t: time-step tensor of size (batch_size,)
        #   x[i] contains image i which has been added noise amount corresponding to t[i]
        # Returns:
        #   noise_pred: noise prediction made from the model, size (batch_size, 3, 32, 32)

        x = self.init_conv(x)

        t = self.time_mlp(time) if exists(self.time_mlp) else None

        h = []

        # downsample
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        # bottleneck
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # upsample
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)

        noise_pred = self.final_conv(x)
        return noise_pred

def extract(a, t, x_shape):
    # Takes a data tensor a and an index tensor t, and returns a new tensor
    # whose i^th element is just a[t[i]]. Note that this will be useful when
    # we would want to choose the alphas or betas corresponding to different
    # indices t's in a batched manner without for loops.
    # Inputs:
    #   a: Tensor, generally of shape (batch_size,)
    #   t: Tensor, generally of shape (batch_size,)
    #   x_shape: Shape of the data, generally (batch_size, 3, 32, 32)
    # Returns:
    #   out: Tensor of shape (batch_size, 1, 1, 1) generally, the number of 1s are
    #         determined by the number of dimensions in x_shape.
    #         out[i] contains a[t[i]]

    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

from diffusion_solution import alphas_betas_sequences_helper

betas, alpha, sqrt_recip_alphas, alphas_cumprod, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, alphas_cumprod_prev, posterior_variance = alphas_betas_sequences_helper(beta_start, beta_end, T)

from diffusion_solution import q_sample

def visualize_diffusion():
    train_dataloader, _ = get_dataloaders()
    batch = next(iter(train_dataloader))
    sample = batch['pixel_values'][3].unsqueeze(0)
    noisy_images = [sample] + [q_sample(sample, torch.tensor([100 * t + 99]), (sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)) for t in range(10)]
    noisy_images = (torch.cat(noisy_images, dim=0) + 1.) * 0.5
    show_image(noisy_images.clamp(0., 1.), nrow=11)

if __name__ == '__main__':
    visualize_diffusion()

from diffusion_solution import p_sample, p_sample_loop

def sample(model, image_size, batch_size=16, channels=3):
    # Returns a sample by running the sampling loop
    with torch.no_grad():
        return p_sample_loop(model, shape=(batch_size, channels, image_size, image_size), timesteps=T, coefficients=(betas, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas, posterior_variance))

from diffusion_solution import p_losses, t_sample

if __name__ == '__main__':
    model = Unet(
        dim=image_size,
        channels=input_channels,
        dim_mults=(1, 4, 16, 64)
    )

    model.to(device)

    optimizer = Adam(model.parameters(), lr=lr)

epochs = 25

if __name__ == '__main__':
    train_dataloader, test_dataloader = get_dataloaders()
    for epoch in range(epochs):
        with tqdm(train_dataloader, unit="batch", leave=False) as tepoch:
            for batch in tepoch:
                tepoch.set_description(f"Epoch: {epoch}")

                optimizer.zero_grad()

                batch_size = batch["pixel_values"].shape[0]
                x = batch["pixel_values"].to(device)

                t = t_sample(T, batch_size, x.device) # Randomly sample timesteps uniformly from [0, T-1]

                loss = p_losses(model, x, t, (sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod))

                loss.backward()
                optimizer.step()

                tepoch.set_postfix(loss=loss.item())

        # Sample and Save Generated Images
        save_image((x + 1.) * 0.5, './results_diffusion/orig.png')
        samples = sample(model, image_size=image_size, batch_size=64, channels=input_channels)
        samples = (torch.Tensor(samples[-1]) + 1.) * 0.5
        save_image(samples, f'./results_diffusion/samples_{epoch}.png')

    show_image(samples)