import warnings
warnings.filterwarnings("ignore")
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
        self.d = 32
        d = self.d

        self.embed_data = nn.Sequential(
            nn.Conv2d(1, d, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(d),
            nn.ReLU(True),

            nn.Conv2d(d, 2*d, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(2*d),
            nn.ReLU(True),

            nn.Conv2d(2*d, 4*d, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(4*d),
            nn.ReLU(True),

            nn.Conv2d(4*d, 8*d, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(8*d),
            nn.ReLU(True),
            
            nn.Flatten(),
        ).to(device)
        # self.embed_condition = nn.Sequential(
        #     nn.Linear(10, 7*7),
        #     # nn.Unflatten(1, (1,7*7)),
        # ).to(device)
        
        self.enc = nn.Sequential(
            nn.Linear(8*d+self.y_dim, d),
            nn.BatchNorm1d(d),
            nn.ReLU(True),

            ).to(device)
        
        self.mu = nn.Linear(d, self.z_dim).to(device)
        self.logvar = nn.Linear(d, self.z_dim).to(device)
        
        self.dec = nn.Sequential(
            nn.Linear(self.z_dim + self.y_dim, 49*8*d),
            nn.Unflatten(1, (8*d, 7, 7)),
            
            nn.ConvTranspose2d(8*d, 4*d, kernel_size = 4, bias = False),
            nn.BatchNorm2d(4*d),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(4*d, 2*d, kernel_size = 4, bias = False),
            nn.BatchNorm2d(2*d),
            nn.ReLU(True),
            
            # reconstruction
            nn.ConvTranspose2d(2*d, 1, kernel_size = 4, stride = 2),
            
            ).to(device)
        
        init_params(self.embed_data)
        # init_params(self.embed_condition)
        init_params(self.enc)
        init_params(self.dec)
        init_params(self.mu)
        init_params(self.logvar)

        self.encoder_trainable = [self.enc, self.embed_data, self.mu, self.logvar]
        self.decoder_trainable = [self.dec]

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
