import torch
import torch.nn as nn
import torch.distributions as dist
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os

from .sampler import multinomial, gaus


class inc_avg:
    def __init__(self):
        self.avg = 0.0
        self.weight = 0

    def append(self, dat, w=1):
        self.weight += w
        self.avg = self.avg + (dat - self.avg) * w / self.weight


def init_params(model):
    for p in model.parameters():
        if p.dim() > 1:
            # nn.init.xavier_normal_(p)
            nn.init.trunc_normal_(p, std=0.01, a=-0.02, b=0.02)
        else:
            nn.init.uniform_(p, 0.1, 0.2)


def reparameterize(mu, logvar):
    # reparameterization trick for probablistic encoder
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


class prob_enc(nn.Module):
    def __init__(self, n, m):
        super(prob_enc, self).__init__()
        self.mu = nn.Linear(n, m)
        self.logvar = nn.Linear(n, m)

    def forward(self, x):
        return reparameterize(self.mu(x), self.logvar(x))


class prob_gauss_mix(nn.Module):
    def __init__(self, n, m, eps):
        super(prob_gauss_mix, self).__init__()
        self.mu = nn.Linear(n, m, bias=False)
        self.eps = eps

    def forward(self, x):
        mu = self.mu(x)
        return mu + self.eps * torch.randn_like(mu)


class prob_mixture_enc(nn.Module):
    def __init__(self, n, m, k=1):
        super(prob_mixture_enc, self).__init__()
        self.k = k
        self.n = n
        self.m = m
        self.code = torch.eye(k)
        self.mu = nn.Linear(n * k, m, bias=False)
        self.logvar = nn.Linear(n * k, m, bias=False)

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.code = self.code.to(*args, **kwargs)
        return self

    def sample_multi(self, n):
        return self.code[torch.randint(self.k, (n,))].repeat(1, self.n)

    def initialize(self):
        self.logvar.weight.data.fill_(-2)
        self.mu.weight.data = torch.repeat_interleave(gaus(self.m, self.n), self.k, dim=1)

    def forward(self, x):
        # Expect input from multinomial distribution
        n = len(x)
        xx = torch.repeat_interleave(x, self.k, dim=1) * self.sample_multi(n)
        return reparameterize(self.mu(xx), self.logvar(xx))


class MixedLayer(nn.Module):
    # Linear map mapping p -> q, r -> s, i -> i
    def __init__(self, p, q, r, s, i, prob=False, mode=1):
        super(MixedLayer, self).__init__()
        self.p = p
        self.q = q
        self.r = r
        self.s = s
        self.i = i
        self.prob = prob
        if self.prob:
            self.mu1 = prob_mixture_enc(p, q, mode)
            self.mu2 = prob_mixture_enc(r, s, mode)
        else:
            self.mu1 = nn.Linear(p, q)
            self.mu2 = nn.Linear(r, s)

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.mu1 = self.mu1.to(*args, **kwargs)
        self.mu2 = self.mu2.to(*args, **kwargs)
        return self

    def initialize(self):
        if self.prob:
            self.mu1.initialize()
            self.mu2.initialize()
        else:
            init_params(self.mu1)
            init_params(self.mu2)

    def forward(self, x):
        return torch.cat(
            (
                self.mu1(x[:, 0 : self.p]),
                self.mu2(x[:, self.p : (self.p + self.r)]),
                x[:, (self.p + self.r) : (self.p + self.r + self.i)],
            ),
            axis=1,
        )


def gather_centroid(cls, dat):
    labs, idxs, counts = cls.unique(dim=0, return_inverse=True, return_counts=True)

    idxs = idxs.view(idxs.size(0), 1).expand(-1, dat.size(1))
    res = torch.zeros((labs.size(0), dat.size(1)), dtype=torch.float).scatter_add_(0, idxs, dat)
    res = res / counts.view(res.size(0), 1)

    return res.gather(0, idxs)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def save_sample_images(save_path, epoch, img_list):
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    fig = plt.figure(figsize=(4, 4))
    for i in range(64):
        plt.subplot(8, 8, i + 1)
        if img_list.shape[1] == 1:
            plt.imshow(np.transpose(img_list[i, :, :, :], (1, 2, 0)), cmap="gray")
        else:
            plt.imshow(np.transpose(img_list[i, :, :, :], (1, 2, 0)))
        plt.axis("off")

    fig.tight_layout(pad=0)
    fig.subplots_adjust(wspace=0.0, hspace=0.0)
    plt.savefig("%s-%03d.png" % (save_path, epoch + 1))


class lap_filter(nn.Module):
    def __init__(self, input_channel=1, device="cpu"):
        super(lap_filter, self).__init__()
        self.filter = nn.Conv2d(1, 1, kernel_size=3, bias=False)
        self.filter.weight = torch.nn.Parameter(
            torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
            .unsqueeze(0)
            .unsqueeze(0)
        )
        self.filter.to(device)
        self.transform = None
        if input_channel == 3:
            self.transform = transforms.functional.rgb_to_grayscale

    def forward(self, x):
        if self.transform is not None:
            return self.filter(self.transform(x))
        return self.filter(x)


def calculate_sharpness(dataset, batch_size=256, labeled=False, device="cpu", num_workers=8):
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers
    )

    if labeled:
        lap = lap_filter(input_channel=dataset[0][0].shape[0], device=device)
        blurr = np.zeros(len(dataset))
        for i, (data, label) in enumerate(dataloader):
            if data.shape[0] == batch_size:
                blurr[(i * batch_size) : ((i + 1) * batch_size)] = (
                    lap(data.to(device)).var(axis=(1, 2, 3)).detach().to("cpu").numpy()
                )
            else:  # last batch
                blurr[(i * batch_size) : len(dataset)] = (
                    lap(data.to(device)).var(axis=(1, 2, 3)).detach().to("cpu").numpy()
                )
    else:
        lap = lap_filter(input_channel=dataset[0].shape[0], device=device)
        blurr = np.zeros(len(dataset))
        for i, data in enumerate(dataloader):
            if data.shape[0] == batch_size:
                blurr[(i * batch_size) : ((i + 1) * batch_size)] = (
                    lap(data.to(device)).var(axis=(1, 2, 3)).detach().to("cpu").numpy()
                )
            else:  # last batch
                blurr[(i * batch_size) : len(dataset)] = (
                    lap(data.to(device)).var(axis=(1, 2, 3)).detach().to("cpu").numpy()
                )

    return blurr


def calculate_sharpness_generator(generator, z_sampler, batch_size=50, repeat=10, device="cpu"):
    blurr = np.empty(batch_size * repeat)
    lap = lap_filter(input_channel=3, device=device)

    for i in range(repeat):
        batch = generator(z_sampler(batch_size).to(device))
        if batch.shape[1] == 1:
            batch = batch.repeat((1, 3, 1, 1))
        blurr[(i * batch_size) : ((i + 1) * batch_size)] = (
            lap(batch.to(device)).var(axis=(1, 2, 3)).detach().to("cpu").numpy()
        )
    return blurr


def make_swiss_roll(num_points, radius_scaling=0.0, num_periods=3, z_max=20.0):
    """
    A quick function to generate swiss roll datasets

    Inputs:
        num_points - how many data points to output, integer
        radius_scaling - the effective "radius" of the spiral will be increased proportionally to z multiplied by this
            constant.  Float
        num_periods - the number of rotations around the origin the spiral will form, float
        z_max - the z values of the data will be uniformly distributed between (0,z_max), float
    Outputs:
        data - a tensor of shape (num_points,3)
    """

    t = np.linspace(0, num_periods * 2.0 * np.pi, num_points)
    x = np.cos(t) * t
    y = np.sin(t) * t
    z = np.random.uniform(low=0.0, high=z_max, size=num_points)

    x *= 1.0 + radius_scaling * z
    y *= 1.0 + radius_scaling * z

    data = np.stack([x, y, z], axis=1)

    return data.astype(np.float32)