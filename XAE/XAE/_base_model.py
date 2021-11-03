import torch
import torch.nn as nn
import torch.optim as optim
from .util import reparameterize

from . import dataset

class AE_abstract(nn.Module):
    def __init__(self, cfg, log, device = 'cpu', verbose = 1):
        super(AE_abstract, self).__init__()
        self.log = log
        if verbose == 1:
            self.log.info('------------------------------------------------------------')
            for key in cfg['train_info']:
                self.log.info('%s : %s' % (key, cfg['train_info'][key]))

            for key in cfg['path_info']:
                self.log.info('%s : %s' % (key, cfg['path_info'][key]))

        self.cfg = cfg

        # Concrete Parts
        self.device = device
        self.z_dim = int(cfg['train_info']['z_dim'])

        data_class = getattr(dataset, cfg['train_info']['train_data'])
        labeled = cfg['train_info'].getboolean('train_data_label')
        self.train_data =  data_class(cfg['path_info']['data_home'], train = True, label = labeled)
        self.test_data = data_class(cfg['path_info']['data_home'], train = False, label = labeled)
        self.prob_enc = cfg['train_info'].getboolean('prob_enc')

        self.batch_size = int(cfg['train_info']['batch_size'])
        self.validate_batch = cfg['train_info'].getboolean('validate')
        if cfg['train_info'].getboolean('replace'):
            it = int(cfg['train_info']['iter_per_epoch'])
            train_sampler = torch.utils.data.RandomSampler(self.train_data, replacement = True, num_samples = self.batch_size * it)
            self.train_generator = torch.utils.data.DataLoader(self.train_data, self.batch_size, num_workers = 5, sampler = train_sampler, pin_memory=True)
        else:
            self.train_generator = torch.utils.data.DataLoader(self.train_data, self.batch_size, num_workers = 5, shuffle = True, pin_memory=True, drop_last=True)

        self.test_generator = torch.utils.data.DataLoader(self.test_data, self.batch_size, num_workers = 5, shuffle = False, pin_memory=True, drop_last=True)
        
        self.save_best = cfg['train_info'].getboolean('save_best')
        self.save_path = cfg['path_info']['save_path']
        self.tensorboard_dir = cfg['path_info']['tb_logs']
        self.save_img_path = cfg['path_info']['save_img_path']
        self.save_state = cfg['path_info']['save_state']
        
        self.encoder_pretrain = cfg['train_info'].getboolean('encoder_pretrain')
        if self.encoder_pretrain:
            self.encoder_pretrain_batch_size = int(cfg['train_info']['encoder_pretrain_batch_size'])
            self.encoder_pretrain_step = int(cfg['train_info']['encoder_pretrain_max_step'])
            self.pretrain_generator = torch.utils.data.DataLoader(self.train_data, self.encoder_pretrain_batch_size, num_workers = 5, shuffle = True, pin_memory=True, drop_last=True)
        
        self.lr = float(cfg['train_info']['lr'])
        self.beta1 = float(cfg['train_info']['beta1'])
        self.lamb = float(cfg['train_info']['lambda'])
        self.lr_schedule = cfg['train_info']['lr_schedule']
        self.num_epoch = int(cfg['train_info']['epoch'])

        # Abstract Parts need overriding
        self.enc = nn.Identity()
        self.mu = nn.Identity()
        self.logvar = nn.Identity()
        self.dec = nn.Identity()

        self.encoder_trainable = [self.enc]
        self.decoder_trainable = [self.dec]

    def encode(self, x):
        if self.prob_enc:
            z = self.enc(x)
            return reparameterize(self.mu(z), self.logvar(z))
        else:
            return self.enc(x)
    
    def decode(self, z):
        return self.dec(z)
        
    def forward(self, x):
        return self.decode(self.encode(x))

    def save(self, dir):
        torch.save(self.state_dict(), dir)

    def load(self, dir = None):
        if dir is None:
            self.load_state_dict(torch.load(self.save_path))
        self.load_state_dict(torch.load(dir))

    def lr_scheduler(self, optimizer, decay = 1.0):
        lamb = lambda e: decay
        if self.lr_schedule is "basic":
            lamb = lambda e: 1.0 / (1.0 + decay * e)
        if self.lr_schedule is "manual":
            lamb = lambda e: decay * 1.0 * (0.5 ** (e >= 30)) * (0.2 ** (e >= 50)) * (0.1 ** (e >= 100))
        return optim.lr_scheduler.MultiplicativeLR(optimizer, lamb)
        
    def pretrain_encoder(self):
        raise NotImplementedError()

    def train(self, resume = False):
        raise NotImplementedError()

class AE_adv_abstract(AE_abstract):
    def __init__(self, cfg, log, device = 'cpu', verbose = 1):
        super(AE_adv_abstract, self).__init__(cfg, log, device, verbose)
        self.lr_adv = float(cfg['train_info']['lr_adv'])
        self.beta1_adv = float(cfg['train_info']['beta1_adv'])

        # Abstract Parts need overriding
        self.disc = nn.Identity()
        self.discriminator_trainable = [self.disc]

    def discriminate(self, z):
        return self.disc(z)