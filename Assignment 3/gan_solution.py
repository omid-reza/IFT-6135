import torch
from torch import nn
from torch.optim import Adam

z_dim = 32
input_channels = 1
device = "cuda" if torch.cuda.is_available() else "cpu"

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

discriminator_optimizer = Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))
generator_optimizer = Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.999))

criterion = nn.BCEWithLogitsLoss()

def discriminator_train(discriminator, generator, real_samples, fake_samples):
  ones = torch.ones(real_samples.size(0), 1, 1, 1, device=device)
  zeros = torch.zeros(fake_samples.size(0), 1, 1, 1, device=device)
  real_output = discriminator(real_samples)
  fake_output = discriminator(fake_samples.detach())
  real_loss = criterion(real_output, ones)
  fake_loss = criterion(fake_output, zeros)
  return real_loss + fake_loss

def generator_train(discriminator, generator, fake_samples):
  ones = torch.ones(fake_samples.size(0), 1, 1, 1, device=device)
  output = discriminator(fake_samples)
  return criterion(output, ones)

def sample(generator, num_samples, noise=None):
  # Takes as input the number of samples and returns that many generated samples
  # Inputs:
  #   num_samples: Scalar denoting the number of samples
  # Returns:
  #   samples: Samples generated; tensor of shape (num_samples, 3, 32, 32)
  if noise is None:
    noise = torch.randn(num_samples, z_dim, 1, 1, device=device)

  with torch.no_grad():
    return generator(noise)

def interpolate(generator, z_1, z_2, n_samples):
  alpha_values = torch.linspace(0, 1, n_samples, device=device).view(-1, 1, 1, 1)
  interpolated_z = alpha_values * z_1 + (1 - alpha_values) * z_2
  with torch.no_grad():
    return generator(interpolated_z)