import os, sys, configparser, logging, argparse
sys.path.append('/'.join(os.getcwd().split('/')[:-2]))
import torch
import torch.nn as nn

from XAE.model import CVAE_abstract
from XAE.logging_daily import logging_daily
from XAE.util import init_params

class CVAE_MNIST(CVAE_abstract):
    def __init__(self, cfg, log, device = 'cpu', verbose = 1):
        super(CVAE_MNIST, self).__init__(cfg, log, device, verbose)
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
            nn.Linear(49*2*d+self.y_dim, d),
            nn.BatchNorm1d(d),
            nn.ReLU(True),

            ).to(device)
        
        self.mu = nn.Linear(d, self.z_dim).to(device)
        self.logvar = nn.Linear(d, self.z_dim).to(device)
        
        self.dec = nn.Sequential(
            nn.Linear(self.z_dim + self.y_dim, 49*2*d),
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
            
            ).to(device)
        
        self.encoder_trainable = [self.enc, self.embed_data, self.mu, self.logvar]
        self.decoder_trainable = [self.dec]

        for net in self.encoder_trainable:
            init_params(net)
        for net in self.decoder_trainable:
            init_params(net)

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
