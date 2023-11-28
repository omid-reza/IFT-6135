"""
Template for Question 3.
@author: Samuel Lavoie
"""
import torch
from q3_sampler import svhn_sampler
from q3_model import Critic, Generator
from q2_solution import lp_reg
from torch import optim
from skimage import io
from skimage import img_as_ubyte
from collections import defaultdict
import math
import os
import numpy as np
import pandas as pd


def combine_images(images):
    total, width, height, channels = images.shape
    rows = int(math.sqrt(total))
    cols = math.ceil(total / rows)
    combined_image = np.zeros(
        (height * rows, width * cols, channels), dtype=images.dtype
    )

    for index, image in enumerate(images):
        i = index // cols
        j = index % cols
        combined_image[
            width * i : width * (i + 1), height * j : height * (j + 1), :
        ] = image
    return combined_image


def make_dirs(experiment_dir):
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "visualizations"), exist_ok=True)


def dl_loop(dl):
    def generator():
        while True:
            for d in dl:
                yield d

    return generator()


if __name__ == "__main__":
    # Example of usage of the code provided and recommended hyper parameters for training GANs.
    data_root = "./"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_iter = 50000  # N training iterations
    n_critic_updates = 5  # N critic updates per generator update
    lp_coeff = 10  # Lipschitz penalty coefficient
    train_batch_size = 64
    test_batch_size = 64
    lr = 1e-4
    beta1 = 0.5
    beta2 = 0.9
    z_dim = 100

    # Configuration
    load_checkpoint = None
    experiment_dir = "training_v1"
    log_step = 50
    viz_step = 100
    save_viz = True
    save_step = 2000
    samples_per_image = 256

    logs = defaultdict(list)
    make_dirs(experiment_dir)
    fixed_noises = torch.randn((samples_per_image, z_dim, 1, 1)).to(device)

    # Initialize
    train_loader, valid_loader, test_loader = svhn_sampler(
        data_root, train_batch_size, test_batch_size
    )
    train_loader = dl_loop(train_loader)

    generator = Generator(z_dim=z_dim).to(device)
    critic = Critic().to(device)

    optim_critic = optim.Adam(critic.parameters(), lr=lr, betas=(beta1, beta2))
    optim_generator = optim.Adam(
        generator.parameters(), lr=lr, betas=(beta1, beta2)
    )

    if load_checkpoint:
        state_dict = torch.load(load_checkpoint)
        critic.load_state_dict(state_dict["critic"])
        generator.load_state_dict(state_dict["generator"])

    # COMPLETE TRAINING PROCEDURE
    for iteration in range(n_iter):
        # train critic
        for critic_iter in range(n_critic_updates):
            images, labels = next(train_loader)
            real_data = images.to(device)
            noise = torch.randn((train_batch_size, z_dim, 1, 1)).to(device)
            generated_data = generator(noise)

            lipschitz_penalty = lp_coeff * lp_reg(
                real_data, generated_data, critic
            )
            w_dist = (critic(real_data) - critic(generated_data)).mean()
            c_loss = -w_dist + lipschitz_penalty

            critic.zero_grad()
            c_loss.backward()
            optim_critic.step()

        # train generator
        noise = torch.randn((train_batch_size, z_dim, 1, 1)).to(device)
        generated_data = generator(noise)
        f_generated = critic(generated_data).mean()
        g_loss = -f_generated

        generator.zero_grad()
        g_loss.backward()
        optim_generator.step()

        # logging
        logs["w_dist"].append(w_dist.item())
        logs["critic_loss"].append(c_loss.item())
        logs["gen_loss"].append(g_loss.item())
        logs["lp"].append(lipschitz_penalty.item())

        if iteration % log_step == 0:
            print("Iteration:", iteration)
            print(f"Wasserstein distance: {logs['w_dist'][-1] : .2f}")
            print(f"Critic loss:          {logs['critic_loss'][-1] : .2f}")
            print(f"Generator loss:       {logs['gen_loss'][-1] : .2f}")
            print("\n")

        # visualization
        if save_viz and iteration % viz_step == 0:
            gen_samples = generator(fixed_noises).cpu().detach().numpy()
            gen_samples = gen_samples.transpose(0, 2, 3, 1)
            gen_samples *= 0.5
            gen_samples += 0.5

            visualization = combine_images(gen_samples)
            save_path = os.path.join(
                experiment_dir, "visualizations", f"viz_{iteration}.jpg"
            )
            io.imsave(save_path, img_as_ubyte(visualization))

        # checkpoint
        if iteration % save_step == 0:
            state_dict = {
                "critic": critic.state_dict(),
                "generator": generator.state_dict(),
            }
            checkpoint_path = os.path.join(
                experiment_dir, "checkpoints", f"ckpt_{iteration}.pth"
            )
            torch.save(state_dict, checkpoint_path)

            csv_path = os.path.join(
                experiment_dir, "logs", f"log_{iteration}.csv"
            )
            csv = pd.DataFrame.from_dict(logs).to_csv(csv_path, index=False)

    # COMPLETE QUALITATIVE EVALUATION
