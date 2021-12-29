import os, sys, configparser, logging, argparse
sys.path.append('/'.join(os.getcwd().split('/')[:-2]))
import torch
import torch.nn as nn

from XAE.model import SSWAE_HSIC_dev
from XAE.logging_daily import logging_daily
from XAE.sampler import multinomial
from XAE.util import init_params, prob_mixture_enc

class SSWAE_HSIC_MNIST(SSWAE_HSIC_dev):
    def __init__(self, cfg, log, device = 'cpu', verbose = 1):
        super(SSWAE_HSIC_MNIST, self).__init__(cfg, log, device, verbose)
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

        self.embed_condition = nn.Sequential(
            nn.Linear(49*2*d, d),
            nn.BatchNorm1d(d),
            nn.ReLU(True),

            nn.Linear(d, self.y_dim),

            nn.Softmax(dim = 1),
        ).to(device)
        
        self.enc = nn.Sequential(
            nn.Linear(49*2*d, d),
            nn.BatchNorm1d(d),
            nn.ReLU(True),

            nn.Linear(d, self.z_dim)
            ).to(device)
        
        self.dec = nn.Sequential(
            nn.Linear(self.y_dim + self.z_dim, 49*2*d),
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

        self.disc = nn.Sequential(
            nn.Linear(self.z_dim, d),
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
        
        self.encoder_trainable = [self.enc, self.embed_data, self.embed_condition]
        self.decoder_trainable = [self.dec]
        self.discriminator_trainable = [self.disc]
        self.pretrain_freeze = [self.embed_data, self.embed_condition]

        for net in self.encoder_trainable:
            init_params(net)
        for net in self.decoder_trainable:
            init_params(net)
        for net in self.discriminator_trainable:
            init_params(net)
            
        # self.embed_data.load_state_dict(torch.load('embed_data_weight.pt'))
        # self.embed_condition.load_state_dict(torch.load('embed_condition_weight.pt'))

    def penalty_loss_mask(self, q, prior_y, mask):
        # mmd = 0.0
        x = q[:,0:self.y_dim]
        n = len(x)
        yy = multinomial(n, self.y_dim, self.device)
        mmd = (self.k(x,x, False) + self.k(yy,yy, False))/(n*(n-1)) - 2*self.k(x,yy, True)/(n*n)

        z = q[:,self.y_dim:]
        qz = self.discriminate(z)
        gan = self.disc_loss(qz, torch.ones_like(qz))

        hsic = self.hsic(x, z)

        return gan, mmd, hsic


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
