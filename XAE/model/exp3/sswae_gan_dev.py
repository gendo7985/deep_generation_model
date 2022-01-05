import os, sys, configparser, logging, argparse
sys.path.append('/'.join(os.getcwd().split('/')[:-2]))
import torch
import torch.nn as nn

from XAE.model import SSWAE_GAN_dev
from XAE.logging_daily import logging_daily
from XAE.sampler import gaus
from XAE.util import init_params, prob_mixture_enc

class SSWAE_GAN_MNIST(SSWAE_GAN_dev):
    def __init__(self, cfg, log, device = 'cpu', verbose = 1):
        super(SSWAE_GAN_MNIST, self).__init__(cfg, log, device, verbose)
        self.d = 64
        d = self.d
        
        self.embed_data = nn.Sequential(
            nn.Conv2d(1, d, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(d),
            nn.ReLU(True),

            nn.Conv2d(d, d, kernel_size = 4, padding = 'same', bias = False),
            nn.BatchNorm2d(d),
            nn.ReLU(True),

            nn.Conv2d(d, 2*d, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(2*d),
            nn.ReLU(True),

            nn.Conv2d(2*d, 2*d, kernel_size = 4, padding = 'same', bias = False),
            nn.BatchNorm2d(2*d),
            nn.ReLU(True),

            nn.Flatten(),
        ).to(device)
        # self.embed_condition = nn.Sequential(
        #     nn.Linear(10, 7*7),
        #     nn.Unflatten(1, (1,7*7)),
        # ).to(device)
        
        self.enc = nn.Sequential(
            nn.Linear(49*2*d, d),
            nn.BatchNorm1d(d),
            nn.ReLU(True),

            nn.Linear(d, self.z_dim)
            ).to(device)

        self.enc2 = nn.Sequential(
            nn.Linear(49*2*d, d),
            nn.BatchNorm1d(d),
            nn.ReLU(True),

            nn.Linear(d, self.yz_dim)
            ).to(device)

        mode = 10
        self.enc_c = prob_mixture_enc(self.y_dim - 1, self.yz_dim, mode).to(device)
        
        self.dec = nn.Sequential(
            nn.Linear(self.yz_dim + self.z_dim, 49*2*d),
            nn.Unflatten(1, (2*d, 7, 7)),
            
            nn.ConvTranspose2d(2*d, d, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(d),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(d, d//2, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(d//2),
            nn.ReLU(True),

            nn.Conv2d(d//2, d//4, kernel_size = 4, padding = 'same', bias = False),
            nn.BatchNorm2d(d//4),
            nn.ReLU(True),
            
            # reconstruction
            nn.Conv2d(d//4, 1, kernel_size = 4, padding = 'same'),
            nn.Tanh(),
            
            ).to(device)

        ff = 10
        self.dec_c = nn.Sequential(
            nn.Linear(self.yz_dim, ff),
            nn.BatchNorm1d(ff),
            nn.ReLU(True),
            nn.Linear(ff, ff),
            nn.BatchNorm1d(ff),
            nn.ReLU(True),
            nn.Linear(ff, self.y_dim),

            # nn.Sigmoid()
            ).to(device)

        self.disc = nn.Sequential(
            nn.Linear(self.yz_dim + self.z_dim, d),
            nn.ReLU(True),

            nn.Linear(d, d),
            nn.ReLU(True),

            nn.Linear(d, d),
            nn.ReLU(True),

            nn.Linear(d, d),
            nn.ReLU(True),

            nn.Linear(d, d),
            nn.ReLU(True),

            nn.Linear(d, 1),
            ).to(device)
        
        self.encoder_trainable = [self.enc, self.enc2, self.enc_c, self.embed_data]
        self.decoder_trainable = [self.dec, self.dec_c]
        self.discriminator_trainable = [self.disc]

        for net in self.encoder_trainable:
            init_params(net)
        for net in self.decoder_trainable:
            init_params(net)
        for net in self.discriminator_trainable:
            init_params(net)
        
        self.enc_c.initialize()
        self.enc_c.to(device)

    def forward(self, x, y):
        return self.decode(torch.cat((self.encode(x,y), self.embed_label(y)), dim = 1))

if __name__ == '__main__':
    is_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if is_cuda else 'cpu')

    parser = argparse.ArgumentParser(description='Training Autoencoders')
    parser.add_argument('--log_info', type=str, help='path of file about log format')
    parser.add_argument('--train_config', type=str, help='path of training configuration file')
    args = parser.parse_args()

    logger = logging_daily(args.log_info)
    log = logger.get_logging()
    log.setLevel(logging.INFO)

    cfg = configparser.ConfigParser()
    cfg.read(args.train_config)

    network = getattr(sys.modules[__name__], cfg['train_info']['model_name'])
    model = network(cfg, log, device = device)
    model.train()
