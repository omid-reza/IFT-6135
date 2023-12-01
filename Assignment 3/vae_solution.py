import torch
import torch.nn as nn


class Encoder(nn.Module):
  def __init__(self, nc, nef, nz, isize, device):
    super(Encoder, self).__init__()

    # Device
    self.device = device

    # Encoder: (nc, isize, isize) -> (nef*8, isize//16, isize//16)
    self.encoder = nn.Sequential(
        nn.Conv2d(nc, nef, 4, 2, padding=1),
        nn.LeakyReLU(0.2, True),
        nn.BatchNorm2d(nef),

        nn.Conv2d(nef, nef * 2, 4, 2, padding=1),
        nn.LeakyReLU(0.2, True),
        nn.BatchNorm2d(nef * 2),

        nn.Conv2d(nef * 2, nef * 4, 4, 2, padding=1),
        nn.LeakyReLU(0.2, True),
        nn.BatchNorm2d(nef * 4),

        nn.Conv2d(nef * 4, nef * 8, 4, 2, padding=1),
        nn.LeakyReLU(0.2, True),
        nn.BatchNorm2d(nef * 8),
    )

  def forward(self, inputs):
    batch_size = inputs.size(0)
    hidden = self.encoder(inputs)
    hidden = hidden.view(batch_size, -1)
    return hidden

class Decoder(nn.Module):
  def __init__(self, nc, ndf, nz, isize):
    super(Decoder, self).__init__()

    # Map the latent vector to the feature map space
    self.ndf = ndf
    self.out_size = isize // 16
    self.decoder_dense = nn.Sequential(
        nn.Linear(nz, ndf * 8 * self.out_size * self.out_size),
        nn.ReLU(True)
    )

    self.decoder_conv = nn.Sequential(
        nn.UpsamplingNearest2d(scale_factor=2),
        nn.Conv2d(ndf * 8, ndf * 4, 3, 1, padding=1),
        nn.LeakyReLU(0.2, True),

        nn.UpsamplingNearest2d(scale_factor=2),
        nn.Conv2d(ndf * 4, ndf * 2, 3, 1, padding=1),
        nn.LeakyReLU(0.2, True),

        nn.UpsamplingNearest2d(scale_factor=2),
        nn.Conv2d(ndf * 2, ndf, 3, 1, padding=1),
        nn.LeakyReLU(0.2, True),

        nn.UpsamplingNearest2d(scale_factor=2),
        nn.Conv2d(ndf, nc, 3, 1, padding=1)
    )

  def forward(self, input):
    batch_size = input.size(0)
    hidden = self.decoder_dense(input).view(
        batch_size, self.ndf * 8, self.out_size, self.out_size)
    output = self.decoder_conv(hidden)
    return output


class DiagonalGaussianDistribution(object):
  # Gaussian Distribution with diagonal covariance matrix
  def __init__(self, mean, logvar=None):
    super(DiagonalGaussianDistribution, self).__init__()
    # Parameters:
    # mean: A tensor representing the mean of the distribution
    # logvar: Optional tensor representing the log of the standard variance
    #         for each of the dimensions of the distribution

    self.mean = mean
    if logvar is None:
        logvar = torch.zeros_like(self.mean)
    self.logvar = torch.clamp(logvar, -30., 20.)

    self.std = torch.exp(0.5 * self.logvar)
    self.var = torch.exp(self.logvar)

  def sample(self, noise=None):
    # Provide a reparameterized sample from the distribution
    # Return: Tensor of the same size as the mean
    if noise is None:
        noise = torch.randn_like(self.mean)
    return self.mean + noise * self.std

  def kl(self):
    # Compute the KL-Divergence between the distribution with the standard normal N(0, I)
    # Return: Tensor of size (batch size,) containing the KL-Divergence for each element in the batch
    return torch.sum(self.var + self.mean**2 - 1 - self.logvar, dim=1)/2

  def nll(self, sample, dims=[1, 2, 3]):
    # Computes the negative log likelihood of the sample under the given distribution
    # Return: Tensor of size (batch size,) containing the log-likelihood for each element in the batch
    return torch.sum((sample - self.mean)**2 / self.var + self.logvar, dim=dims)/2

  def mode(self):
    # Returns the mode of the distribution
    return self.mean


class VAE(nn.Module):
  def __init__(self, in_channels=3, decoder_features=32, encoder_features=32, z_dim=100, input_size=32, device=torch.device("cuda:0")):
    super(VAE, self).__init__()

    self.z_dim = z_dim
    self.in_channels = in_channels
    self.device = device

    # Encode the Input
    self.encoder = Encoder(nc=in_channels,
                            nef=encoder_features,
                            nz=z_dim,
                            isize=input_size,
                            device=device
                            )

    # Map the encoded feature map to the latent vector of mean, (log)variance
    out_size = input_size // 16
    self.mean = nn.Linear(encoder_features * 8 * out_size * out_size, z_dim)
    self.logvar = nn.Linear(encoder_features * 8 * out_size * out_size, z_dim)

    # Decode the Latent Representation
    self.decoder = Decoder(nc=in_channels,
                           ndf=decoder_features,
                           nz=z_dim,
                           isize=input_size
                           )

  def encode(self, x):
    hidden = self.encoder(x)
    hidden = hidden.view(hidden.size(0), -1)
    return DiagonalGaussianDistribution(self.mean(hidden), self.logvar(hidden))

  def decode(self, z):
    return DiagonalGaussianDistribution(self.decoder(z.view(z.size(0), -1)))

  def sample(self, batch_size, noise=None):
    prior = DiagonalGaussianDistribution(torch.zeros(batch_size, self.z_dim).to(self.device))
    z = prior.sample(noise)
    recon = self.decode(z.view(z.size(0), -1))
    return recon.mode()

  def log_likelihood(self, x, K=100):
    # Approximate the log-likelihood of the data using Importance Sampling
    # Inputs:
    #   x: Data sample tensor of shape (batch_size, 3, 32, 32)
    #   K: Number of samples to use to approximate p_\theta(x)
    # Returns:
    #   ll: Log likelihood of the sample x in the VAE model using K samples
    #       Size: (batch_size,)
    posterior = self.encode(x)
    prior = DiagonalGaussianDistribution(torch.zeros_like(posterior.mean))

    log_likelihood = torch.zeros(x.shape[0], K).to(self.device)
    for i in range(K):
      z = posterior.sample()
      recon = self.decode(z.view(z.size(0), -1))
      log_likelihood[:, i] = prior.nll(z)-recon.nll(x)
      del z, recon

    ll = torch.logsumexp(log_likelihood, dim=1) - torch.log(torch.tensor(K, dtype=torch.float32, device=self.device))
    return ll

  def forward(self, x, noise=None):
    posterior = self.encode(x)
    latent_z = posterior.sample(noise)
    recon = self.decode(latent_z.view(latent_z.size(0), -1))
    return recon.mode(), recon.nll(x), posterior.kl()


def interpolate(model, z_1, z_2, n_samples):
  lengths = torch.linspace(0., 1., n_samples).unsqueeze(1).to(z_1.device)
  z = lengths * z_1 + (1 - lengths) * z_2
  return model.decode(z).mode()