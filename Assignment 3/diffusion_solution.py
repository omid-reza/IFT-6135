import torch
import torch.nn.functional as F

from tqdm import tqdm

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


def alphas_betas_sequences_helper(beta_start, beta_end, T):
    def linear_beta_schedule(beta_start, beta_end, timesteps):
        return torch.linspace(beta_start, beta_end, timesteps)

    betas = linear_beta_schedule(beta_start, beta_end, T)
    alphas = 1 - betas
    sqrt_recip_alphas = 1 / torch.sqrt(alphas)
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod =  torch.sqrt(1 - alphas_cumprod)
    alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=betas.device), alphas_cumprod[:-1]])
    posterior_variance = betas * (1 - alphas_cumprod_prev)/(1 - alphas_cumprod)
    return betas, alphas, sqrt_recip_alphas, alphas_cumprod, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, alphas_cumprod_prev, posterior_variance

def q_sample(x_start, t, coefficients, noise=None):
    if noise is None:
      noise = torch.randn_like(x_start)
    sqrt_alphas_cumprod_t = extract(coefficients[0], t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(coefficients[1], t, x_start.shape)
    x_noisy = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    return x_noisy

def p_sample(model, x, t, t_index, coefficients,  noise=None):
    # Given the denoising model, batched input x, and time-step t, returns a slightly denoised sample at time-step t-1
    # Inputs:
    #   model: The denoising (parameterized noise) model
    #   x: Batched noisy input at time t; size (batch_size, 3, 32, 32)
    #   t: Batched time steps; size (batch_size,)
    #   t_index: Single time-step, whose batched version is present in t
    #   coefficients: 4-tuple
    # Returns:
    #   sample: A sample from the distribution p_\theta(x_{t-1} | x_t); mode if t=0
    with torch.no_grad():
        betas_t = extract(coefficients[0], t, x.shape)                         # WRITE CODE HERE: Similar to q_sample, extract betas for specific t's
        sqrt_one_minus_alphas_cumprod_t = extract(coefficients[1], t, x.shape) # WRITE CODE HERE: Same as above, but for sqrt_one_minus_alphas_cumprod
        sqrt_recip_alphas_t = extract(coefficients[2], t, x.shape)            # WRITE CODE HERE: Same as above, but for sqrt_recip_alphas

        p_mean = sqrt_recip_alphas_t * (x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t)

        if t_index == 0:
            sample = p_mean                       # WRITE CODE HERE: Set the sample as the mode
        else:
            posterior_variance_t = extract(coefficients[3], t, x.shape)
            if noise is None:
                noise = torch.randn_like(x)
            sample = p_mean + torch.sqrt(posterior_variance_t) * noise

        return sample

def p_sample_loop(model, shape, timesteps, T, coefficients, noise=None):
    # Given the model, and the shape of the image, returns a sample from the data distribution by running through the backward diffusion process.
    # Inputs:
    #   model: The denoising model
    #   shape: Shape of the samples; set as (batch_size, 3, 32, 32)
    #   noise: (timesteps+1, batch_size, 3, 32, 32)
    # Returns:
    #   imgs: Samples obtained, as well as intermediate denoising steps, of shape (T, batch_size, 3, 32, 32)
    with torch.no_grad():
        b = shape[0]
        # Start from pure noise (x_T)
        img = torch.randn(shape, device=model.device) if noise is None else noise[0]
        imgs = []

        for i in tqdm(reversed(range(0, timesteps)), desc='Sampling', total=T, leave=False):
            img = None
            imgs.append(img.cpu())

        return torch.stack(imgs)

def p_losses(denoise_model, x_start, t, coefficients, noise=None):
    noise = torch.randn_like(x_start) if noise is None else noise
    x_noisy = q_sample(x_start=x_start, t=t, coefficients=coefficients, noise=noise)
    predicted_noise = denoise_model(x_noisy, t)
    loss = F.smooth_l1_loss(noise, predicted_noise)
    return loss


def t_sample(timesteps, batch_size, device):
    ts = torch.randint(0, timesteps, (batch_size,), device=device)
    return ts