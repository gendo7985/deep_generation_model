import time
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from torch.utils.tensorboard import SummaryWriter
    import tensorflow as tf
    import tensorboard as tb
    tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
except ImportError:
    pass

from .util import inc_avg, save_sample_images, reparameterize
from . import dataset, sampler
from ._base_model import XAE_abstract, XAE_adv_abstract, CXAE_abstract, CXAE_adv_abstract

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

class WAE_MMD_abstract(XAE_abstract):
    def __init__(self, cfg, log, device = 'cpu', verbose = 1):
        super(WAE_MMD_abstract, self).__init__(cfg, log, device, verbose)
        self.loss = nn.MSELoss()

    def k(self, x, y, diag = True):
        stat = 0.
        for scale in [.1, .2, .5, 1., 2., 5., 10.]:
            C = scale*2*self.z_dim*2
            kernel = (C/(C + (x.unsqueeze(0) - y.unsqueeze(1)).pow(2).sum(dim = 2)))
            if diag:
                stat += kernel.sum()
            else:
                stat += kernel.sum() - kernel.diag().sum()
        return stat
        
    def penalty_loss(self, x, y, n):
        return (self.k(x,x, False) + self.k(y,y, False))/(n*(n-1)) - 2*self.k(x,y, True)/(n*n)

class CWAE_MMD_abstract(CXAE_abstract):
    def __init__(self, cfg, log, device = 'cpu', verbose = 1):
        super(CWAE_MMD_abstract, self).__init__(cfg, log, device, verbose)
        self.loss = nn.MSELoss()

    def k(self, x, y, diag = True):
        stat = 0.
        for scale in [.1, .2, .5, 1., 2., 5., 10.]:
            C = scale*2*self.z_dim*2
            kernel = (C/(C + (x.unsqueeze(0) - y.unsqueeze(1)).pow(2).sum(dim = 2)))
            if diag:
                stat += kernel.sum()
            else:
                stat += kernel.sum() - kernel.diag().sum()
        return stat
        
    def penalty_loss(self, x, y, n):
        return (self.k(x,x, False) + self.k(y,y, False))/(n*(n-1)) - 2*self.k(x,y, True)/(n*n)

class WAE_GAN_abstract(XAE_adv_abstract):
    def __init__(self, cfg, log, device = 'cpu', verbose = 1):
        super(WAE_GAN_abstract, self).__init__(cfg, log, device, verbose)
        self.loss = nn.MSELoss()
        self.disc_loss = nn.BCEWithLogitsLoss()

    def penalty_loss(self, q):
        qz = self.discriminate(q)
        return self.disc_loss(qz, torch.ones_like(qz))

    def adv_loss(self, p, q):
        pz = self.discriminate(p)
        qz = self.discriminate(q)
        return self.disc_loss(pz, torch.ones_like(pz)) + self.disc_loss(qz, torch.zeros_like(qz))

class CWAE_GAN_abstract(CXAE_adv_abstract):
    def __init__(self, cfg, log, device = 'cpu', verbose = 1):
        super(CWAE_GAN_abstract, self).__init__(cfg, log, device, verbose)
        self.loss = nn.MSELoss()
        self.disc_loss = nn.BCEWithLogitsLoss()

    def penalty_loss(self, q):
        qz = self.discriminate(q)
        return self.disc_loss(qz, torch.ones_like(qz))

    def adv_loss(self, p, q):
        pz = self.discriminate(p)
        qz = self.discriminate(q)
        return self.disc_loss(pz, torch.ones_like(pz)) + self.disc_loss(qz, torch.zeros_like(qz))
        
class VAE_abstract(XAE_abstract):
    def __init__(self, cfg, log, device = 'cpu', verbose = 1):
        super(VAE_abstract, self).__init__(cfg, log, device, verbose)
        self.loss = nn.BCEWithLogitsLoss(reduction = 'sum')
        # self.log2pi = torch.log(2.0 * np.pi)
        self.mu = nn.Identity()
        self.logvar = nn.Identity()

    def encode(self, x):
        xx = self.enc(x)
        return reparameterize(self.mu(xx), self.logvar(xx))

    def decode(self, x):
        return self.dec(x).tanh()

    def mu_and_logvar(self, x):
        xx = self.enc(x)
        return self.mu(xx), self.logvar(xx)
        
    def penalty_loss(self, mu, logvar):
        return (-.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(dim = 1)).mean()

    def pretrain_encoder(self):
        raise NotImplementedError()

    def train(self, resume = False):
        train_main_list = []
        train_penalty_list = []
        test_main_list = []
        test_penalty_list = []

        for net in self.encoder_trainable:
            net.train()
        for net in self.decoder_trainable:
            net.train()
            
        if self.encoder_pretrain:
            self.pretrain_encoder()
            self.log.info('Pretraining Ended!')
            
        if len(self.tensorboard_dir) > 0:
            self.writer = SummaryWriter(self.tensorboard_dir)
            
        optimizer = optim.Adam(sum([list(net.parameters()) for net in self.encoder_trainable], []) + sum([list(net.parameters()) for net in self.decoder_trainable], []), lr = self.lr, betas = (self.beta1, 0.999))

        start_epoch = 0
        scheduler = self.lr_scheduler(optimizer)

        if resume:
            checkpoint = torch.load(self.save_state)
            start_epoch = checkpoint['epoch']
            self.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if len(self.lr_schedule) > 0:
                scheduler.load_state_dict(checkpoint['scheduler'])

        trep = len(self.train_generator)
        ftrep = len(str(trep))
        ttep = len(self.test_generator)
        fttep = len(str(ttep))
        fep = len(str(self.num_epoch))
        
        self.log.info('------------------------------------------------------------')
        self.log.info('Training Start!')
        start_time = time.time()
        
        for epoch in range(start_epoch, self.num_epoch):
            # train_step
            train_loss_main = inc_avg()
            train_loss_penalty = inc_avg()
            
            for i, data in enumerate(self.train_generator):
                for net in self.encoder_trainable:
                    net.zero_grad()
                for net in self.decoder_trainable:
                    net.zero_grad()

                n = len(data)
                x = data.to(self.device)
                mu, logvar = self.mu_and_logvar(x)
                recon = self.dec(reparameterize(mu, logvar))
                
                loss = self.main_loss((recon + 1.0)*0.5, (x + 1.0)*0.5) / n
                if self.lamb > 0:
                    penalty = self.penalty_loss(mu, logvar)
                    obj = loss + self.lamb * penalty
                else:
                    obj = loss
                obj.backward()
                optimizer.step()
                
                train_loss_main.append(loss.item(), n)
                if self.lamb > 0:
                    train_loss_penalty.append(penalty.item(), n)
                
                print(f'[{i+1:0{ftrep}}/{trep}]  train_main: {train_loss_main.avg:.4f}  train_penalty: {train_loss_penalty.avg:.4f}', end = "\r")

            train_main_list.append(train_loss_main.avg)
            train_penalty_list.append(train_loss_penalty.avg)

            if len(self.tensorboard_dir) > 0:
                self.writer.add_scalar('train/main', train_loss_main.avg, epoch)
                self.writer.add_scalar('train/penalty', train_loss_penalty.avg, epoch)
                if self.cfg['train_info'].getboolean('histogram'):
                    for param_tensor in self.state_dict():
                        self.writer.add_histogram(param_tensor, self.state_dict()[param_tensor].detach().to('cpu').numpy().flatten(), epoch)
            
            # validation_step
            test_loss_main = inc_avg()
            test_loss_penalty = inc_avg()

            if self.validate_batch:
                for i, data in enumerate(self.test_generator):

                    n = len(data)
                    x = data.to(self.device)
                    mu, logvar = self.mu_and_logvar(x)
                    recon = self.dec(reparameterize(mu, logvar))

                    test_loss_main.append(self.main_loss((recon + 1.0)*0.5, (x + 1.0)*0.5).item() / n, n)
                    if self.lamb > 0:
                        test_loss_penalty.append(self.penalty_loss(mu, logvar).item(), n)
                    print(f'[{i+1:0{fttep}}/{ttep}]  test_main: {test_loss_main.avg:.4f}  test_penalty: {test_loss_penalty.avg:.4f}', end = "\r")

                test_main_list.append(test_loss_main.avg)
                test_penalty_list.append(test_loss_penalty.avg)
                
                self.log.info(f'[{epoch + 1:0{fep}}/{self.num_epoch}]  train_main: {train_loss_main.avg:.6e}  train_penalty: {train_loss_penalty.avg:.6e}  test_main: {test_loss_main.avg:.6e}  test_penalty: {test_loss_penalty.avg:.6e}')

                # Additional test set
                data = next(iter(self.test_generator))

                x = data.to(self.device)
                fake_latent = self.encode(x).detach()
                recon = self.decode(fake_latent).detach()
                prior_z = self.generate_prior(len(data)).detach()

                if len(self.tensorboard_dir) > 0:
                    self.writer.add_scalar('test/main', test_loss_main.avg, epoch)
                    self.writer.add_scalar('test/penalty', test_loss_penalty.avg, epoch)

                    if self.lamb > 0:
                        # Embedding
                        for_embed1 = fake_latent.to('cpu').numpy()
                        for_embed2 = prior_z.to('cpu').numpy()
                        label = ['fake']*len(for_embed1) + ['prior']*len(for_embed2)
                        self.writer.add_embedding(np.concatenate((for_embed1, for_embed2)), metadata = label, global_step = epoch)

                        # Sample Generation
                        test_dec = self.decode(prior_z).detach().to('cpu').numpy()
                        self.writer.add_images('generated_sample', (test_dec[0:32])*0.5 + 0.5, epoch)

                    # Reconstruction
                    self.writer.add_images('reconstruction', (np.concatenate((x.to('cpu').numpy()[0:16], recon.to('cpu').numpy()[0:16])))*0.5 + 0.5, epoch)
                    self.writer.flush()

                if len(self.save_img_path) > 0:
                    save_sample_images('%s/recon' % self.save_img_path, epoch, (np.concatenate((x.detach().to('cpu').numpy()[0:32], recon.detach().to('cpu').numpy()[0:32])))*0.5 + 0.5)
                    plt.close()
                    if self.lamb > 0:
                        # Sample Generation
                        test_dec = self.decode(prior_z).detach().to('cpu').numpy()
                        save_sample_images('%s/gen' % self.save_img_path, epoch, (test_dec[0:64])*0.5 + 0.5)
                        plt.close()
                
                if self.save_best:
                    obj = test_loss_main.avg + self.lamb * test_loss_penalty.avg
                    if self.best_obj[1] > obj:
                        self.best_obj[0] = epoch + 1
                        self.best_obj[1] = obj
                        self.save(self.save_path)
                        self.log.info("model saved, obj: %.6e" % obj)
                else:
                    self.save(self.save_path)
                    # self.log.info("model saved at: %s" % self.save_path)
                
            scheduler.step()

            if len(self.save_state) > 0:
                save_dict = {
                    'epoch':epoch + 1,
                    'model_state_dict':self.state_dict(),
                    'optimizer_state_dict':optimizer.state_dict()
                } 
                if len(self.lr_schedule) > 0:
                    save_dict['scheduler'] = scheduler.state_dict()
                torch.save(save_dict, self.save_state)
            
        if not self.validate_batch:
            self.save(self.save_path)

        self.log.info('Training Finished!')
        self.log.info("Elapsed time: %.3fs" % (time.time() - start_time))

        if len(self.tensorboard_dir) > 0:
            self.writer.close()

        self.hist = np.array([train_main_list , train_penalty_list, test_main_list, test_penalty_list])

class CVAE_abstract(XAE_abstract):
    def __init__(self, cfg, log, device = 'cpu', verbose = 1):
        super(CVAE_abstract, self).__init__(cfg, log, device, verbose)
        self.loss = nn.BCEWithLogitsLoss(reduction = 'sum')
        self.y_sampler = getattr(sampler, cfg['train_info']['y_sampler']) # generate prior
        self.y_dim = int(cfg['train_info']['y_dim'])

        self.mu = nn.Identity()
        self.logvar = nn.Identity()
        self.embed_data = nn.Identity()
        self.embed_condition = nn.Identity()

    def generate_prior(self, n):
        return self.z_sampler(n, self.z_dim, device = self.device), self.y_sampler(n, self.y_dim, device = self.device)

    def encode(self, x, y):
        xx = self.enc(torch.cat((self.embed_data(x), self.embed_condition(y)), dim = 1))
        return reparameterize(self.mu(xx), self.logvar(xx))

    def decode(self, x, y):
        return self.dec(torch.cat((x, y), dim = 1)).tanh()

    def forward(self, x, y):
        return self.decode(self.encode(x,y), y)

    def mu_and_logvar(self, x, y):
        xx = self.enc(torch.cat((self.embed_data(x), self.embed_condition(y)), dim = 1))
        return self.mu(xx), self.logvar(xx)
        
    def penalty_loss(self, mu, logvar):
        return (-.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(dim = 1)).mean()

    def pretrain_encoder(self):
        raise NotImplementedError()

    def train(self, resume = False):
        train_main_list = []
        train_penalty_list = []
        test_main_list = []
        test_penalty_list = []

        for net in self.encoder_trainable:
            net.train()
        for net in self.decoder_trainable:
            net.train()
            
        if self.encoder_pretrain:
            self.pretrain_encoder()
            self.log.info('Pretraining Ended!')
            
        if len(self.tensorboard_dir) > 0:
            self.writer = SummaryWriter(self.tensorboard_dir)
            
        optimizer = optim.Adam(sum([list(net.parameters()) for net in self.encoder_trainable], []) + sum([list(net.parameters()) for net in self.decoder_trainable], []), lr = self.lr, betas = (self.beta1, 0.999))

        start_epoch = 0
        scheduler = self.lr_scheduler(optimizer)

        if resume:
            checkpoint = torch.load(self.save_state)
            start_epoch = checkpoint['epoch']
            self.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if len(self.lr_schedule) > 0:
                scheduler.load_state_dict(checkpoint['scheduler'])

        trep = len(self.train_generator)
        ftrep = len(str(trep))
        ttep = len(self.test_generator)
        fttep = len(str(ttep))
        fep = len(str(self.num_epoch))
        
        self.log.info('------------------------------------------------------------')
        self.log.info('Training Start!')
        start_time = time.time()
        
        for epoch in range(start_epoch, self.num_epoch):
            # train_step
            train_loss_main = inc_avg()
            train_loss_penalty = inc_avg()
            
            for i, (data, condition) in enumerate(self.train_generator):
                for net in self.encoder_trainable:
                    net.zero_grad()
                for net in self.decoder_trainable:
                    net.zero_grad()

                n = len(data)
                x = data.to(self.device)
                y = condition.to(self.device)
                mu, logvar = self.mu_and_logvar(x, y)
                recon = self.dec(torch.cat((reparameterize(mu, logvar), y), dim = 1))
                
                loss = self.main_loss((recon + 1.0)*0.5, (x + 1.0)*0.5) / n
                if self.lamb > 0:
                    penalty = self.penalty_loss(mu, logvar)
                    obj = loss + self.lamb * penalty
                else:
                    obj = loss
                obj.backward()
                optimizer.step()
                
                train_loss_main.append(loss.item(), n)
                if self.lamb > 0:
                    train_loss_penalty.append(penalty.item(), n)
                
                print(f'[{i+1:0{ftrep}}/{trep}]  train_main: {train_loss_main.avg:.4f}  train_penalty: {train_loss_penalty.avg:.4f}', end = "\r")

            train_main_list.append(train_loss_main.avg)
            train_penalty_list.append(train_loss_penalty.avg)

            if len(self.tensorboard_dir) > 0:
                self.writer.add_scalar('train/main', train_loss_main.avg, epoch)
                self.writer.add_scalar('train/penalty', train_loss_penalty.avg, epoch)
                if self.cfg['train_info'].getboolean('histogram'):
                    for param_tensor in self.state_dict():
                        self.writer.add_histogram(param_tensor, self.state_dict()[param_tensor].detach().to('cpu').numpy().flatten(), epoch)
            
            # validation_step
            test_loss_main = inc_avg()
            test_loss_penalty = inc_avg()

            if self.validate_batch:
                for i, (data, condition) in enumerate(self.test_generator):

                    n = len(data)
                    x = data.to(self.device)
                    y = condition.to(self.device)
                    mu, logvar = self.mu_and_logvar(x, y)
                    recon = self.dec(torch.cat((reparameterize(mu, logvar), y), dim = 1))

                    test_loss_main.append(self.main_loss((recon + 1.0)*0.5, (x + 1.0)*0.5).item() / n, n)
                    if self.lamb > 0:
                        test_loss_penalty.append(self.penalty_loss(mu, logvar).item(), n)
                    print(f'[{i+1:0{fttep}}/{ttep}]  test_main: {test_loss_main.avg:.4f}  test_penalty: {test_loss_penalty.avg:.4f}', end = "\r")

                test_main_list.append(test_loss_main.avg)
                test_penalty_list.append(test_loss_penalty.avg)
                
                self.log.info(f'[{epoch + 1:0{fep}}/{self.num_epoch}]  train_main: {train_loss_main.avg:.6e}  train_penalty: {train_loss_penalty.avg:.6e}  test_main: {test_loss_main.avg:.6e}  test_penalty: {test_loss_penalty.avg:.6e}')

                # Additional test set
                data, condition = next(iter(self.test_generator))

                x = data.to(self.device)
                y = condition.to(self.device)
                fake_latent = self.encode(x, y)
                recon = self.decode(fake_latent, y).detach()
                prior_z, prior_y = self.generate_prior(len(data))

                if len(self.tensorboard_dir) > 0:
                    self.writer.add_scalar('test/main', test_loss_main.avg, epoch)
                    self.writer.add_scalar('test/penalty', test_loss_penalty.avg, epoch)

                    if self.lamb > 0:
                        # Embedding
                        for_embed1 = torch.cat((fake_latent, y), dim = 1).detach().to('cpu').numpy()
                        for_embed2 = torch.cat((prior_z, prior_y), dim = 1).detach().to('cpu').numpy()
                        label = ['fake']*len(for_embed1) + ['prior']*len(for_embed2)
                        self.writer.add_embedding(np.concatenate((for_embed1, for_embed2)), metadata = label, global_step = epoch)

                        # Sample Generation
                        test_dec = self.decode(prior_z, prior_y).detach().to('cpu').numpy()
                        self.writer.add_images('generated_sample', (test_dec[0:32])*0.5 + 0.5, epoch)

                    # Reconstruction
                    self.writer.add_images('reconstruction', (np.concatenate((x.to('cpu').numpy()[0:16], recon.to('cpu').numpy()[0:16])))*0.5 + 0.5, epoch)
                    self.writer.flush()

                if len(self.save_img_path) > 0:
                    save_sample_images('%s/recon' % self.save_img_path, epoch, (np.concatenate((x.detach().to('cpu').numpy()[0:32], recon.detach().to('cpu').numpy()[0:32])))*0.5 + 0.5)
                    plt.close()
                    if self.lamb > 0:
                        # Sample Generation
                        test_dec = self.decode(prior_z, prior_y).detach().to('cpu').numpy()
                        save_sample_images('%s/gen' % self.save_img_path, epoch, (test_dec[0:64])*0.5 + 0.5)
                        plt.close()
                
                if self.save_best:
                    obj = test_loss_main.avg + self.lamb * test_loss_penalty.avg
                    if self.best_obj[1] > obj:
                        self.best_obj[0] = epoch + 1
                        self.best_obj[1] = obj
                        self.save(self.save_path)
                        self.log.info("model saved, obj: %.6e" % obj)
                else:
                    self.save(self.save_path)
                    # self.log.info("model saved at: %s" % self.save_path)
                
            scheduler.step()

            if len(self.save_state) > 0:
                save_dict = {
                    'epoch':epoch + 1,
                    'model_state_dict':self.state_dict(),
                    'optimizer_state_dict':optimizer.state_dict()
                } 
                if len(self.lr_schedule) > 0:
                    save_dict['scheduler'] = scheduler.state_dict()
                torch.save(save_dict, self.save_state)
            
        if not self.validate_batch:
            self.save(self.save_path)

        self.log.info('Training Finished!')
        self.log.info("Elapsed time: %.3fs" % (time.time() - start_time))

        if len(self.tensorboard_dir) > 0:
            self.writer.close()

        self.hist = np.array([train_main_list , train_penalty_list, test_main_list, test_penalty_list])

class SSWAE_MMD_abstract(CWAE_MMD_abstract):
    def __init__(self, cfg, log, device = 'cpu', verbose = 1):
        super(SSWAE_MMD_abstract, self).__init__(cfg, log, device, verbose)
        self.yz_dim = int(cfg['train_info']['yz_dim'])
        self.lamb2 = float(cfg['train_info']['lambda2'])

        data_class = getattr(dataset, cfg['train_info']['train_data'])
        labeled = cfg['train_info'].getboolean('train_data_label')

        labeled_class = cfg['train_info']['labeled_class'].replace(' ', '').split(',')
        labeled_class = [int(i) for i in labeled_class]
        unlabeled_class = cfg['train_info']['unlabeled_class'].replace(' ', '').split(',')
        unlabeled_class = [int(i) for i in unlabeled_class]
        test_class = cfg['train_info']['test_class'].replace(' ', '').split(',')
        test_class = [int(i) for i in test_class]

        self.batch_size1 = int(cfg['train_info']['batch_size1'])
        self.batch_size2 = int(cfg['train_info']['batch_size2'])
        batch_size = max(self.batch_size1, self.batch_size2)

        self.train_data1 = data_class(cfg['path_info']['data_home2'], train = True, label = labeled, aux = [labeled_class, []])
        self.train_generator1 = torch.utils.data.DataLoader(self.train_data1, self.batch_size1, num_workers = 5, shuffle = True, pin_memory=True, drop_last=True)

        self.train_data2 = data_class(cfg['path_info']['data_home2'], train = True, label = labeled, aux = [labeled_class, unlabeled_class])
        self.train_generator2 = torch.utils.data.DataLoader(self.train_data2, self.batch_size2, num_workers = 5, shuffle = True, pin_memory=True, drop_last=True)

        self.test_data = data_class(cfg['path_info']['data_home2'], train = False, label = labeled, aux = [labeled_class, unlabeled_class + test_class])
        self.test_generator = torch.utils.data.DataLoader(self.test_data, batch_size, num_workers = 5, shuffle = True, pin_memory=True, drop_last=False)

        self.enc2 = nn.Identity()
        self.enc_c = nn.Identity()
        self.dec_c = nn.Identity()
        try:
            w = float(cfg['train_info']['classification_weight'])
            self.bce = nn.BCEWithLogitsLoss(weight = torch.cat((torch.ones(self.y_dim - 1)/(self.y_dim - 1)*(1-w),torch.Tensor([w]))).to(self.device))
        except:
            self.bce = nn.BCEWithLogitsLoss()

    def generate_prior(self, n):
        return self.z_sampler(n, self.yz_dim + self.z_dim, device = self.device)

    def encode_s(self, x):
        xx = self.embed_data(x)
        return torch.cat((self.enc2(xx), self.enc(xx)), dim = 1)

    def encode(self, x, y):
        return torch.cat((self.enc_c(y[:,:-1]), self.enc(self.embed_data(x))), dim = 1)
    
    def decode(self, z):
        return self.dec(z)

    def decode_c(self, z):
        return self.dec_c(z[:,0:self.yz_dim])
        
    def forward(self, x, y):
        return self.decode(self.encode(x, y))

    def classify_loss(self, recon, y):
        return self.bce(recon, y)

    def k(self, x, y, diag = True):
        stat = 0.
        for scale in [.1, .2, .5, 1., 2., 5., 10.]:
            C = scale*2*(self.z_dim+self.yz_dim)*2
            kernel = (C/(C + (x.unsqueeze(0) - y.unsqueeze(1)).pow(2).sum(dim = 2)))
            if diag:
                stat += kernel.sum()
            else:
                stat += kernel.sum() - kernel.diag().sum()
        return stat
        
    def penalty_loss(self, x, y, n):
        return (self.k(x,x, False) + self.k(y,y, False))/(n*(n-1)) - 2*self.k(x,y, True)/(n*n)

    def train(self, resume = False):
        train_main_list = []
        train_main2_list = []
        train_penalty_list = []
        test_main_list = []
        test_main2_list = []
        test_penalty_list = []

        for net in self.encoder_trainable:
            net.train()
        for net in self.decoder_trainable:
            net.train()
            
        if self.encoder_pretrain:
            self.pretrain_encoder()
            self.log.info('Pretraining Ended!')
            
        if len(self.tensorboard_dir) > 0:
            self.writer = SummaryWriter(self.tensorboard_dir)
            
        optimizer = optim.Adam(sum([list(net.parameters()) for net in self.encoder_trainable], []) + sum([list(net.parameters()) for net in self.decoder_trainable], []), lr = self.lr, betas = (self.beta1, 0.999))

        start_epoch = 0
        scheduler = self.lr_scheduler(optimizer)

        if resume:
            checkpoint = torch.load(self.save_state)
            start_epoch = checkpoint['epoch']
            self.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if len(self.lr_schedule) > 0:
                scheduler.load_state_dict(checkpoint['scheduler'])

        iter_per_epoch = min(len(self.train_generator1), len(self.train_generator2))
        n1 = self.batch_size1
        n2 = self.batch_size2

        trep = iter_per_epoch
        ftrep = len(str(iter_per_epoch))
        ttep = len(self.test_generator)
        fttep = len(str(ttep))
        fep = len(str(self.num_epoch))

        self.log.info('------------------------------------------------------------')
        self.log.info('Training Start!')
        start_time = time.time()

        for epoch in range(start_epoch, self.num_epoch):
            # train_step
            train_loss_main = inc_avg()
            train_loss_main2 = inc_avg()
            train_loss_penalty = inc_avg()

            tr1 = iter(self.train_generator1)
            tr2 = iter(self.train_generator2)

            for i in range(iter_per_epoch):
                for net in self.encoder_trainable:
                    net.zero_grad()
                for net in self.decoder_trainable:
                    net.zero_grad()
                
                # with label
                data, condition = next(tr1)

                x1 = data.to(self.device)
                y1 = condition.to(self.device)
                fake_latent1 = self.encode(x1, y1)
                recon = self.decode(fake_latent1)
                recon_y = self.decode_c(fake_latent1)
                
                loss1 = self.main_loss(x1, recon)
                loss2 = self.classify_loss(recon_y, y1)
                loss = loss1 + self.lamb2 * loss2

                # without label
                data, condition = next(tr2)

                x2 = data.to(self.device)
                y2 = condition.to(self.device)
                fake_latent2 = self.encode_s(x2)
                recon2 = self.decode(fake_latent2)
                recon_y2 = self.decode_c(fake_latent2)

                los1 = self.main_loss(x2, recon2)
                los2 = self.classify_loss(recon_y2, y2)
                los = los1 + self.lamb2 * los2

                fake_latent = torch.cat((fake_latent1, fake_latent2), dim = 0)
                prior_z = self.generate_prior(n1 + n2)

                if self.lamb > 0:
                    penalty = self.penalty_loss(fake_latent, prior_z, n1+n2)
                    obj = loss *(n1/(n1+n2))  + los *(n2/(n1+n2)) + self.lamb * penalty
                else:
                    obj = loss *(n1/(n1+n2))  + los *(n2/(n1+n2))

                obj.backward()
                optimizer.step()
                
                train_loss_main.append((loss1 *(n1/(n1+n2))  + los1 *(n1/(n1+n2))).item(), n1+n2)
                train_loss_main2.append((loss2 *(n1/(n1+n2))  + los2 *(n1/(n1+n2))).item(), n1+n2)
                if self.lamb > 0:
                    train_loss_penalty.append(penalty.item(), n1+n2)
                
                print(f'[{i+1:0{ftrep}}/{trep}]  train_main: {train_loss_main.avg:.4f}  train_main2: {train_loss_main2.avg:.4f}  train_penalty: {train_loss_penalty.avg:.4f}', end = "\r")

            train_main_list.append(train_loss_main.avg)
            train_main2_list.append(train_loss_main2.avg)
            train_penalty_list.append(train_loss_penalty.avg)

            if len(self.tensorboard_dir) > 0:
                self.writer.add_scalar('train/main', train_loss_main.avg, epoch)
                self.writer.add_scalar('train/penalty', train_loss_penalty.avg, epoch)
                if self.cfg['train_info'].getboolean('histogram'):
                    for param_tensor in self.state_dict():
                        self.writer.add_histogram(param_tensor, self.state_dict()[param_tensor].detach().to('cpu').numpy().flatten(), epoch)
            
            # validation_step
            test_loss_main = inc_avg()
            test_loss_main2 = inc_avg()
            test_loss_penalty = inc_avg()

            if self.validate_batch:
                for i, (data, condition) in enumerate(self.test_generator):
                    n = len(data)

                    # test without label
                    prior_z = self.generate_prior(n)
                    x = data.to(self.device)
                    y = condition.to(self.device)
                    fake_latent = self.encode_s(x)
                    recon = self.decode(fake_latent)
                    recon_y = self.decode_c(fake_latent)

                    loss1 = self.main_loss(x, recon)
                    loss2 = self.classify_loss(recon_y, y)

                    test_loss_main.append(loss1.item(), n)
                    test_loss_main2.append(loss2.item(), n)
                    if self.lamb > 0:
                        test_loss_penalty.append(self.penalty_loss(fake_latent, prior_z, n).item(), n)
                    print(f'[{i+1:0{fttep}}/{ttep}]  test_main: {test_loss_main.avg:.4f}  test_main2: {test_loss_main2.avg:.4f}  test_penalty: {test_loss_penalty.avg:.4f}', end = "\r")

                test_main_list.append(test_loss_main.avg)
                test_main2_list.append(test_loss_main2.avg)
                test_penalty_list.append(test_loss_penalty.avg)
                self.log.info(f'[{epoch + 1:0{fep}}/{self.num_epoch}]  train_main: {train_loss_main.avg:.6e}  train_penalty: {train_loss_penalty.avg:.6e}  train_main2: {train_loss_main2.avg:.4f}  train_penalty: {train_loss_penalty.avg:.6e}  test_main: {test_loss_main.avg:.6e}  test_main2: {test_loss_main2.avg:.6e}  test_penalty: {test_loss_penalty.avg:.6e}')

                # Additional test set
                data, condition = next(iter(self.test_generator))

                n = len(data)
                prior_z = self.generate_prior(n)
                x = data.to(self.device)
                # y = condition.to(self.device)
                fake_latent = self.encode_s(x)
                recon = self.decode(fake_latent)

                if len(self.tensorboard_dir) > 0:
                    self.writer.add_scalar('test/main', test_loss_main.avg, epoch)
                    self.writer.add_scalar('test/penalty', test_loss_penalty.avg, epoch)

                    if self.lamb > 0:
                        # Embedding
                        for_embed1 = fake_latent.detach().to('cpu').numpy()
                        for_embed2 = prior_z.detach().to('cpu').numpy()
                        label = ['fake']*len(for_embed1) + ['prior']*len(for_embed2)
                        self.writer.add_embedding(np.concatenate((for_embed1, for_embed2)), metadata = label, global_step = epoch)

                        # Sample Generation
                        test_dec = self.decode(prior_z).detach().to('cpu').numpy()
                        self.writer.add_images('generated_sample', (test_dec[0:32])*0.5 + 0.5, epoch)

                    # Reconstruction
                    self.writer.add_images('reconstruction', (np.concatenate((x.detach().to('cpu').numpy()[0:16], recon.detach().to('cpu').numpy()[0:16])))*0.5 + 0.5, epoch)
                    self.writer.flush()

                if len(self.save_img_path) > 0:
                    save_sample_images('%s/recon' % self.save_img_path, epoch, (np.concatenate((x.detach().to('cpu').numpy()[0:32], recon.detach().to('cpu').numpy()[0:32])))*0.5 + 0.5)
                    plt.close()
                    if self.lamb > 0:
                        # Sample Generation
                        test_dec = self.decode(prior_z).detach().to('cpu').numpy()
                        save_sample_images('%s/gen' % self.save_img_path, epoch, (test_dec[0:64])*0.5 + 0.5)
                        plt.close()
                
                if self.save_best:
                    obj = test_loss_main.avg + self.lamb * test_loss_penalty.avg
                    if self.best_obj[1] > obj:
                        self.best_obj[0] = epoch + 1
                        self.best_obj[1] = obj
                        self.save(self.save_path)
                        self.log.info("model saved, obj: %.6e" % obj)
                else:
                    self.save(self.save_path)
                    # self.log.info("model saved at: %s" % self.save_path)
                
            scheduler.step()

            if len(self.save_state) > 0:
                save_dict = {
                    'epoch':epoch + 1,
                    'model_state_dict':self.state_dict(),
                    'optimizer_state_dict':optimizer.state_dict()
                } 
                if len(self.lr_schedule) > 0:
                    save_dict['scheduler'] = scheduler.state_dict()
                torch.save(save_dict, self.save_state)
            
        if not self.validate_batch:
            self.save(self.save_path)

        self.log.info('Training Finished!')
        self.log.info("Elapsed time: %.3fs" % (time.time() - start_time))

        if len(self.tensorboard_dir) > 0:
            self.writer.close()

        self.hist = np.array([train_main_list, train_main2_list, train_penalty_list, test_main_list, test_main2_list, test_penalty_list])

class SSWAE_GAN_abstract(CWAE_GAN_abstract):
    def __init__(self, cfg, log, device = 'cpu', verbose = 1):
        super(SSWAE_GAN_abstract, self).__init__(cfg, log, device, verbose)
        self.yz_dim = int(cfg['train_info']['yz_dim'])
        self.lamb2 = float(cfg['train_info']['lambda2'])
        self.disc_loss = nn.BCEWithLogitsLoss()
        
        data_class = getattr(dataset, cfg['train_info']['train_data'])
        labeled = cfg['train_info'].getboolean('train_data_label')

        labeled_class = cfg['train_info']['labeled_class'].replace(' ', '').split(',')
        labeled_class = [int(i) for i in labeled_class]
        unlabeled_class = cfg['train_info']['unlabeled_class'].replace(' ', '').split(',')
        unlabeled_class = [int(i) for i in unlabeled_class]
        test_class = cfg['train_info']['test_class'].replace(' ', '').split(',')
        test_class = [int(i) for i in test_class]

        self.batch_size1 = int(cfg['train_info']['batch_size1'])
        self.batch_size2 = int(cfg['train_info']['batch_size2'])
        batch_size = max(self.batch_size1, self.batch_size2)

        self.train_data1 = data_class(cfg['path_info']['data_home2'], train = True, label = labeled, aux = [labeled_class, []])
        self.train_generator1 = torch.utils.data.DataLoader(self.train_data1, self.batch_size1, num_workers = 5, shuffle = True, pin_memory=True, drop_last=True)

        self.train_data2 = data_class(cfg['path_info']['data_home2'], train = True, label = labeled, aux = [labeled_class, unlabeled_class])
        self.train_generator2 = torch.utils.data.DataLoader(self.train_data2, self.batch_size2, num_workers = 5, shuffle = True, pin_memory=True, drop_last=True)

        self.test_data = data_class(cfg['path_info']['data_home2'], train = False, label = labeled, aux = [labeled_class, unlabeled_class + test_class])
        self.test_generator = torch.utils.data.DataLoader(self.test_data, batch_size, num_workers = 5, shuffle = True, pin_memory=True, drop_last=False)

        self.enc2 = nn.Identity()
        self.enc_c = nn.Identity()
        self.dec_c = nn.Identity()
        try:
            w = float(cfg['train_info']['classification_weight'])
            self.bce = nn.BCEWithLogitsLoss(weight = torch.cat((torch.ones(self.y_dim - 1)/(self.y_dim - 1)*(1-w),torch.Tensor([w]))).to(self.device))
        except:
            self.bce = nn.BCEWithLogitsLoss()

    def generate_prior(self, n):
        return self.z_sampler(n, self.yz_dim + self.z_dim, device = self.device)

    def encode_s(self, x):
        xx = self.embed_data(x)
        return torch.cat((self.enc2(xx), self.enc(xx)), dim = 1)

    def encode(self, x, y):
        return torch.cat((self.enc_c(y[:,:-1]), self.enc(self.embed_data(x))), dim = 1)
    
    def decode(self, z):
        return self.dec(z)

    def decode_c(self, z):
        return self.dec_c(z[:,0:self.yz_dim])
        
    def forward(self, x, y):
        return self.decode(self.encode(x, y))

    def classify_loss(self, recon, y):
        return self.bce(recon, y)

    def train(self, resume = False):
        train_main_list = []
        train_main2_list = []
        train_penalty_list = []
        test_main_list = []
        test_main2_list = []
        test_penalty_list = []

        for net in self.encoder_trainable:
            net.train()
        for net in self.decoder_trainable:
            net.train()
            
        if self.encoder_pretrain:
            self.pretrain_encoder()
            self.log.info('Pretraining Ended!')
            
        if len(self.tensorboard_dir) > 0:
            self.writer = SummaryWriter(self.tensorboard_dir)
            
        optimizer_main = optim.Adam(sum([list(net.parameters()) for net in self.encoder_trainable], []) + sum([list(net.parameters()) for net in self.decoder_trainable], []), lr = self.lr, betas = (self.beta1, 0.999))
        optimizer_adv = optim.Adam(sum([list(net.parameters()) for net in self.discriminator_trainable], []), lr = self.lr_adv, betas = (self.beta1_adv, 0.999))

        start_epoch = 0
        scheduler_main = self.lr_scheduler(optimizer_main)
        scheduler_adv = self.lr_scheduler(optimizer_adv)

        if resume:
            checkpoint = torch.load(self.save_state)
            start_epoch = checkpoint['epoch']
            self.load_state_dict(checkpoint['model_state_dict'])
            optimizer_main.load_state_dict(checkpoint['optimizer_main_state_dict'])
            optimizer_adv.load_state_dict(checkpoint['optimizer_adv_state_dict'])
            if len(self.lr_schedule) > 0:
                scheduler_main.load_state_dict(checkpoint['scheduler_main'])
                scheduler_adv.load_state_dict(checkpoint['scheduler_adv'])
        
        iter_per_epoch = min(len(self.train_generator1), len(self.train_generator2))
        n1 = self.batch_size1
        n2 = self.batch_size2

        trep = iter_per_epoch
        ftrep = len(str(iter_per_epoch))
        ttep = len(self.test_generator)
        fttep = len(str(ttep))
        fep = len(str(self.num_epoch))
        
        self.log.info('------------------------------------------------------------')
        self.log.info('Training Start!')
        start_time = time.time()
        
        for epoch in range(start_epoch, self.num_epoch):
            # train_step
            train_loss_main = inc_avg()
            train_loss_main2 = inc_avg()
            train_loss_penalty = inc_avg()

            tr1 = iter(self.train_generator1)
            tr2 = iter(self.train_generator2)

            for i in range(iter_per_epoch):
                # with label
                data1, condition1 = next(tr1)
                data2, condition2 = next(tr2)
                prior_z = self.generate_prior(n1+n2)

                if self.lamb > 0:
                    for net in self.discriminator_trainable:
                        net.zero_grad()

                    x1 = data1.to(self.device)
                    y1 = condition1.to(self.device)
                    fake_latent1 = self.encode(x1, y1)

                    x2 = data2.to(self.device)
                    fake_latent2 = self.encode_s(x2)

                    fake_latent = torch.cat((fake_latent1, fake_latent2), dim = 0)

                    adv = self.adv_loss(prior_z, fake_latent)
                    obj_adv = self.lamb * adv
                    obj_adv.backward()
                    optimizer_adv.step()

                for net in self.encoder_trainable:
                    net.zero_grad()
                for net in self.decoder_trainable:
                    net.zero_grad()

                # with label

                x1 = data1.to(self.device)
                y1 = condition1.to(self.device)
                fake_latent1 = self.encode(x1, y1)
                recon = self.decode(fake_latent1)
                recon_y = self.decode_c(fake_latent1)
                
                loss1 = self.main_loss(x1, recon)
                loss2 = self.classify_loss(recon_y, y1)
                loss = loss1 + self.lamb2 * loss2

                # without label

                x2 = data2.to(self.device)
                y2 = condition2.to(self.device)
                fake_latent2 = self.encode_s(x2)
                recon2 = self.decode(fake_latent2)
                recon_y2 = self.decode_c(fake_latent2)

                los1 = self.main_loss(x2, recon2)
                los2 = self.classify_loss(recon_y2, y2)
                los = los1 + self.lamb2 * los2

                fake_latent = torch.cat((fake_latent1, fake_latent2), dim = 0)

                if self.lamb > 0:
                    penalty = self.penalty_loss(fake_latent)
                    obj = loss *(n1/(n1+n2))  + los *(n2/(n1+n2)) + self.lamb * penalty
                else:
                    obj = loss *(n1/(n1+n2))  + los *(n2/(n1+n2))

                obj.backward()
                optimizer_main.step()
                
                train_loss_main.append((loss1 *(n1/(n1+n2)) + los1 *(n2/(n1+n2))).item(), n1+n2)
                train_loss_main2.append((loss2 *(n1/(n1+n2)) + los2 *(n2/(n1+n2))).item(), n1+n2)
                if self.lamb > 0:
                    train_loss_penalty.append(penalty.item(), n1+n2)
                
                print(f'[{i+1:0{ftrep}}/{trep}]  train_main: {train_loss_main.avg:.4f}  train_main2: {train_loss_main2.avg:.4f}  train_penalty: {train_loss_penalty.avg:.4f}', end = "\r")

            train_main_list.append(train_loss_main.avg)
            train_main2_list.append(train_loss_main2.avg)
            train_penalty_list.append(train_loss_penalty.avg)

            if len(self.tensorboard_dir) > 0:
                self.writer.add_scalar('train/main', train_loss_main.avg, epoch)
                self.writer.add_scalar('train/penalty', train_loss_penalty.avg, epoch)
                if self.cfg['train_info'].getboolean('histogram'):
                    for param_tensor in self.state_dict():
                        self.writer.add_histogram(param_tensor, self.state_dict()[param_tensor].detach().to('cpu').numpy().flatten(), epoch)
            
            # validation_step
            test_loss_main = inc_avg()
            test_loss_main2 = inc_avg()
            test_loss_penalty = inc_avg()

            if self.validate_batch:
                for i, (data, condition) in enumerate(self.test_generator):
                    n = len(data)
                    # test without label
                    x = data.to(self.device)
                    y = condition.to(self.device)
                    fake_latent = self.encode_s(x)
                    recon = self.decode(fake_latent)
                    recon_y = self.decode_c(fake_latent)

                    loss1 = self.main_loss(x, recon)
                    loss2 = self.classify_loss(recon_y, y)

                    test_loss_main.append(loss1.item(), n)
                    test_loss_main2.append(loss2.item(), n)
                    if self.lamb > 0:
                        test_loss_penalty.append(self.penalty_loss(fake_latent).item(), n)
                    print(f'[{i+1:0{fttep}}/{ttep}]  test_main: {test_loss_main.avg:.4f}  test_main2: {test_loss_main2.avg:.4f}  test_penalty: {test_loss_penalty.avg:.4f}', end = "\r")

                test_main_list.append(test_loss_main.avg)
                test_main2_list.append(test_loss_main2.avg)
                test_penalty_list.append(test_loss_penalty.avg)
                self.log.info(f'[{epoch + 1:0{fep}}/{self.num_epoch}]  train_main: {train_loss_main.avg:.6e}  train_penalty: {train_loss_penalty.avg:.6e}  train_main2: {train_loss_main2.avg:.4f}  train_penalty: {train_loss_penalty.avg:.6e}  test_main: {test_loss_main.avg:.6e}  test_main2: {test_loss_main2.avg:.6e}  test_penalty: {test_loss_penalty.avg:.6e}')

                # Additional test set
                data, condition = next(iter(self.test_generator))

                n = len(data)
                prior_z = self.generate_prior(n)
                x = data.to(self.device)
                # y = condition.to(self.device)
                fake_latent = self.encode_s(x)
                recon = self.decode(fake_latent)

                if len(self.tensorboard_dir) > 0:
                    self.writer.add_scalar('test/main', test_loss_main.avg, epoch)
                    self.writer.add_scalar('test/penalty', test_loss_penalty.avg, epoch)

                    if self.lamb > 0:
                        # Embedding
                        for_embed1 = fake_latent.detach().to('cpu').numpy()
                        for_embed2 = prior_z.detach().to('cpu').numpy()
                        label = ['fake']*len(for_embed1) + ['prior']*len(for_embed2)
                        self.writer.add_embedding(np.concatenate((for_embed1, for_embed2)), metadata = label, global_step = epoch)

                        # Sample Generation
                        test_dec = self.decode(prior_z).detach().to('cpu').numpy()
                        self.writer.add_images('generated_sample', (test_dec[0:32])*0.5 + 0.5, epoch)

                    # Reconstruction
                    self.writer.add_images('reconstruction', (np.concatenate((x.detach().to('cpu').numpy()[0:16], recon.detach().to('cpu').numpy()[0:16])))*0.5 + 0.5, epoch)
                    self.writer.flush()

                if len(self.save_img_path) > 0:
                    save_sample_images('%s/recon' % self.save_img_path, epoch, (np.concatenate((x.detach().to('cpu').numpy()[0:32], recon.detach().to('cpu').numpy()[0:32])))*0.5 + 0.5)
                    plt.close()
                    if self.lamb > 0:
                        # Sample Generation
                        test_dec = self.decode(prior_z).detach().to('cpu').numpy()
                        save_sample_images('%s/gen' % self.save_img_path, epoch, (test_dec[0:64])*0.5 + 0.5)
                        plt.close()
                
                if self.save_best:
                    obj = test_loss_main.avg + self.lamb * test_loss_penalty.avg
                    if self.best_obj[1] > obj:
                        self.best_obj[0] = epoch + 1
                        self.best_obj[1] = obj
                        self.save(self.save_path)
                        self.log.info("model saved, obj: %.6e" % obj)
                else:
                    self.save(self.save_path)
                    # self.log.info("model saved at: %s" % self.save_path)
                
            scheduler_main.step()
            if self.lamb > 0:
                scheduler_adv.step()

            if len(self.save_state) > 0:
                save_dict = {
                    'epoch':epoch + 1,
                    'model_state_dict':self.state_dict(),
                    'optimizer_main_state_dict':optimizer_main.state_dict(),
                    'optimizer_adv_state_dict':optimizer_adv.state_dict()
                } 
                if len(self.lr_schedule) > 0:
                    save_dict['scheduler_main'] = scheduler_main.state_dict()
                    save_dict['scheduler_adv'] = scheduler_adv.state_dict()
                torch.save(save_dict, self.save_state)
            
        if not self.validate_batch:
            self.save(self.save_path)

        self.log.info('Training Finished!')
        self.log.info("Elapsed time: %.3fs" % (time.time() - start_time))

        if len(self.tensorboard_dir) > 0:
            self.writer.close()

        self.hist = np.array([train_main_list, train_main2_list, train_penalty_list, test_main_list, test_main2_list, test_penalty_list])

class SSWAE_MMD_dev(CWAE_MMD_abstract):
    def __init__(self, cfg, log, device = 'cpu', verbose = 1):
        super(SSWAE_MMD_dev, self).__init__(cfg, log, device, verbose)
        self.yz_dim = int(cfg['train_info']['yz_dim'])
        self.lamb2 = float(cfg['train_info']['lambda2'])

        data_class = getattr(dataset, cfg['train_info']['train_data'])
        labeled = cfg['train_info'].getboolean('train_data_label')

        labeled_class = cfg['train_info']['labeled_class'].replace(' ', '').split(',')
        labeled_class = [int(i) for i in labeled_class]
        unlabeled_class = cfg['train_info']['unlabeled_class'].replace(' ', '').split(',')
        unlabeled_class = [int(i) for i in unlabeled_class]
        test_class = cfg['train_info']['test_class'].replace(' ', '').split(',')
        test_class = [int(i) for i in test_class]

        # self.batch_size1 = int(cfg['train_info']['batch_size1'])
        self.batch_size2 = int(cfg['train_info']['batch_size2'])
        # batch_size = max(self.batch_size1, self.batch_size2)

        # self.train_data1 = data_class(cfg['path_info']['data_home2'], train = True, label = labeled, aux = [labeled_class, []])
        # self.train_generator1 = torch.utils.data.DataLoader(self.train_data1, self.batch_size1, num_workers = 5, shuffle = True, pin_memory=True, drop_last=True)

        self.train_data2 = data_class(cfg['path_info']['data_home2'], train = True, label = labeled, aux = [labeled_class, unlabeled_class])
        self.train_generator2 = torch.utils.data.DataLoader(self.train_data2, self.batch_size2, num_workers = 5, shuffle = True, pin_memory=True, drop_last=True)

        self.test_data = data_class(cfg['path_info']['data_home2'], train = False, label = labeled, aux = [labeled_class, unlabeled_class + test_class])
        self.test_generator = torch.utils.data.DataLoader(self.test_data, self.batch_size2, num_workers = 5, shuffle = True, pin_memory=True, drop_last=False)

        self.enc2 = nn.Identity()
        # self.enc_c = nn.Identity()
        self.dec_c = nn.Identity()
        try:
            w = float(cfg['train_info']['classification_weight'])
            self.bce = nn.BCEWithLogitsLoss(weight = torch.cat((torch.ones(self.y_dim - 1)/(self.y_dim - 1)*(1-w),torch.Tensor([w]))).to(self.device))
        except:
            self.bce = nn.BCEWithLogitsLoss()

    def generate_prior(self, n):
        return self.z_sampler(n, self.yz_dim + self.z_dim, device = self.device)

    def encode_s(self, x):
        xx = self.embed_data(x)
        return torch.cat((self.enc2(xx), self.enc(xx)), dim = 1)

    # def encode(self, x, y):
    #     return torch.cat((self.enc_c(y[:,:-1]), self.enc(self.embed_data(x))), dim = 1)
    
    def decode(self, z):
        return self.dec(z)

    def decode_c(self, z):
        return self.dec_c(z[:,0:self.yz_dim])
        
    def forward(self, x, y):
        return self.decode(self.encode(x, y))

    def classify_loss(self, recon, y):
        return self.bce(recon, y)

    def k(self, x, y, diag = True):
        stat = 0.
        for scale in [.1, .2, .5, 1., 2., 5., 10.]:
            C = scale*2*(self.z_dim+self.yz_dim)*2
            kernel = (C/(C + (x.unsqueeze(0) - y.unsqueeze(1)).pow(2).sum(dim = 2)))
            if diag:
                stat += kernel.sum()
            else:
                stat += kernel.sum() - kernel.diag().sum()
        return stat
        
    def penalty_loss(self, x, y, n):
        return (self.k(x,x, False) + self.k(y,y, False))/(n*(n-1)) - 2*self.k(x,y, True)/(n*n)

    def train(self, resume = False):
        train_main_list = []
        train_main2_list = []
        train_penalty_list = []
        test_main_list = []
        test_main2_list = []
        test_penalty_list = []

        for net in self.encoder_trainable:
            net.train()
        for net in self.decoder_trainable:
            net.train()
            
        if self.encoder_pretrain:
            self.pretrain_encoder()
            self.log.info('Pretraining Ended!')
            
        if len(self.tensorboard_dir) > 0:
            self.writer = SummaryWriter(self.tensorboard_dir)
            
        optimizer = optim.Adam(sum([list(net.parameters()) for net in self.encoder_trainable], []) + sum([list(net.parameters()) for net in self.decoder_trainable], []), lr = self.lr, betas = (self.beta1, 0.999))

        start_epoch = 0
        scheduler = self.lr_scheduler(optimizer)

        if resume:
            checkpoint = torch.load(self.save_state)
            start_epoch = checkpoint['epoch']
            self.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if len(self.lr_schedule) > 0:
                scheduler.load_state_dict(checkpoint['scheduler'])

        trep = len(self.train_generator2)
        ftrep = len(str(trep))
        ttep = len(self.test_generator)
        fttep = len(str(ttep))
        fep = len(str(self.num_epoch))

        self.log.info('------------------------------------------------------------')
        self.log.info('Training Start!')
        start_time = time.time()
        iter_per_epoch = len(self.train_generator2)

        for epoch in range(start_epoch, self.num_epoch):
            # train_step
            train_loss_main = inc_avg()
            train_loss_main2 = inc_avg()
            train_loss_penalty = inc_avg()

            for i, (data, condition) in enumerate(self.train_generator2):
                n = len(data)
                for net in self.encoder_trainable:
                    net.zero_grad()
                for net in self.decoder_trainable:
                    net.zero_grad()

                x = data.to(self.device)
                y = condition.to(self.device)
                fake_latent = self.encode_s(x)
                recon = self.decode(fake_latent)
                recon_y = self.decode_c(fake_latent)

                los1 = self.main_loss(x, recon)
                los2 = self.classify_loss(recon_y, y)
                los = los1 + self.lamb2 * los2

                prior_z = self.generate_prior(n)

                if self.lamb > 0:
                    penalty = self.penalty_loss(fake_latent, prior_z, n)
                    obj = los + self.lamb * penalty
                else:
                    obj = los

                obj.backward()
                optimizer.step()
                
                train_loss_main.append(los1.item(), n)
                train_loss_main2.append(los2.item(), n)
                if self.lamb > 0:
                    train_loss_penalty.append(penalty.item(), n)
                
                print(f'[{i+1:0{ftrep}}/{trep}]  train_main: {train_loss_main.avg:.4f}  train_main2: {train_loss_main2.avg:.4f}  train_penalty: {train_loss_penalty.avg:.4f}', end = "\r")

            train_main_list.append(train_loss_main.avg)
            train_main2_list.append(train_loss_main2.avg)
            train_penalty_list.append(train_loss_penalty.avg)

            if len(self.tensorboard_dir) > 0:
                self.writer.add_scalar('train/main', train_loss_main.avg, epoch)
                self.writer.add_scalar('train/penalty', train_loss_penalty.avg, epoch)
                if self.cfg['train_info'].getboolean('histogram'):
                    for param_tensor in self.state_dict():
                        self.writer.add_histogram(param_tensor, self.state_dict()[param_tensor].detach().to('cpu').numpy().flatten(), epoch)
            
            # validation_step
            test_loss_main = inc_avg()
            test_loss_main2 = inc_avg()
            test_loss_penalty = inc_avg()

            if self.validate_batch:
                for i, (data, condition) in enumerate(self.test_generator):
                    n = len(data)

                    # test without label
                    prior_z = self.generate_prior(n)
                    x = data.to(self.device)
                    y = condition.to(self.device)
                    fake_latent = self.encode_s(x)
                    recon = self.decode(fake_latent)
                    recon_y = self.decode_c(fake_latent)

                    loss1 = self.main_loss(x, recon)
                    loss2 = self.classify_loss(recon_y, y)

                    test_loss_main.append(loss1.item(), n)
                    test_loss_main2.append(loss2.item(), n)
                    if self.lamb > 0:
                        test_loss_penalty.append(self.penalty_loss(fake_latent, prior_z, n).item(), n)
                    print(f'[{i+1:0{fttep}}/{ttep}]  test_main: {test_loss_main.avg:.4f}  test_main2: {test_loss_main2.avg:.4f}  test_penalty: {test_loss_penalty.avg:.4f}', end = "\r")

                test_main_list.append(test_loss_main.avg)
                test_main2_list.append(test_loss_main2.avg)
                test_penalty_list.append(test_loss_penalty.avg)
                self.log.info(f'[{epoch + 1:0{fep}}/{self.num_epoch}]  train_main: {train_loss_main.avg:.6e}  train_penalty: {train_loss_penalty.avg:.6e}  train_main2: {train_loss_main2.avg:.4f}  train_penalty: {train_loss_penalty.avg:.6e}  test_main: {test_loss_main.avg:.6e}  test_main2: {test_loss_main2.avg:.6e}  test_penalty: {test_loss_penalty.avg:.6e}')

                # Additional test set
                data, condition = next(iter(self.test_generator))

                n = len(data)
                prior_z = self.generate_prior(n)
                x = data.to(self.device)
                # y = condition.to(self.device)
                fake_latent = self.encode_s(x)
                recon = self.decode(fake_latent)

                if len(self.tensorboard_dir) > 0:
                    self.writer.add_scalar('test/main', test_loss_main.avg, epoch)
                    self.writer.add_scalar('test/penalty', test_loss_penalty.avg, epoch)

                    if self.lamb > 0:
                        # Embedding
                        for_embed1 = fake_latent.detach().to('cpu').numpy()
                        for_embed2 = prior_z.detach().to('cpu').numpy()
                        label = ['fake']*len(for_embed1) + ['prior']*len(for_embed2)
                        self.writer.add_embedding(np.concatenate((for_embed1, for_embed2)), metadata = label, global_step = epoch)

                        # Sample Generation
                        test_dec = self.decode(prior_z).detach().to('cpu').numpy()
                        self.writer.add_images('generated_sample', (test_dec[0:32])*0.5 + 0.5, epoch)

                    # Reconstruction
                    self.writer.add_images('reconstruction', (np.concatenate((x.detach().to('cpu').numpy()[0:16], recon.detach().to('cpu').numpy()[0:16])))*0.5 + 0.5, epoch)
                    self.writer.flush()

                if len(self.save_img_path) > 0:
                    save_sample_images('%s/recon' % self.save_img_path, epoch, (np.concatenate((x.detach().to('cpu').numpy()[0:32], recon.detach().to('cpu').numpy()[0:32])))*0.5 + 0.5)
                    plt.close()
                    if self.lamb > 0:
                        # Sample Generation
                        test_dec = self.decode(prior_z).detach().to('cpu').numpy()
                        save_sample_images('%s/gen' % self.save_img_path, epoch, (test_dec[0:64])*0.5 + 0.5)
                        plt.close()
                
                if self.save_best:
                    obj = test_loss_main.avg + self.lamb * test_loss_penalty.avg
                    if self.best_obj[1] > obj:
                        self.best_obj[0] = epoch + 1
                        self.best_obj[1] = obj
                        self.save(self.save_path)
                        self.log.info("model saved, obj: %.6e" % obj)
                else:
                    self.save(self.save_path)
                    # self.log.info("model saved at: %s" % self.save_path)
                
            scheduler.step()

            if len(self.save_state) > 0:
                save_dict = {
                    'epoch':epoch + 1,
                    'model_state_dict':self.state_dict(),
                    'optimizer_state_dict':optimizer.state_dict()
                } 
                if len(self.lr_schedule) > 0:
                    save_dict['scheduler'] = scheduler.state_dict()
                torch.save(save_dict, self.save_state)
            
        if not self.validate_batch:
            self.save(self.save_path)

        self.log.info('Training Finished!')
        self.log.info("Elapsed time: %.3fs" % (time.time() - start_time))

        if len(self.tensorboard_dir) > 0:
            self.writer.close()

        self.hist = np.array([train_main_list, train_main2_list, train_penalty_list, test_main_list, test_main2_list, test_penalty_list])

class SSWAE_GAN_dev(CWAE_GAN_abstract):
    def __init__(self, cfg, log, device = 'cpu', verbose = 1):
        super(SSWAE_GAN_dev, self).__init__(cfg, log, device, verbose)
        self.yz_dim = int(cfg['train_info']['yz_dim'])
        self.lamb2 = float(cfg['train_info']['lambda2'])
        self.disc_loss = nn.BCEWithLogitsLoss()
        
        data_class = getattr(dataset, cfg['train_info']['train_data'])
        labeled = cfg['train_info'].getboolean('train_data_label')

        labeled_class = cfg['train_info']['labeled_class'].replace(' ', '').split(',')
        labeled_class = [int(i) for i in labeled_class]
        unlabeled_class = cfg['train_info']['unlabeled_class'].replace(' ', '').split(',')
        unlabeled_class = [int(i) for i in unlabeled_class]
        test_class = cfg['train_info']['test_class'].replace(' ', '').split(',')
        test_class = [int(i) for i in test_class]

        # self.batch_size1 = int(cfg['train_info']['batch_size1'])
        self.batch_size2 = int(cfg['train_info']['batch_size2'])
        batch_size = self.batch_size2

        # self.train_data1 = data_class(cfg['path_info']['data_home2'], train = True, label = labeled, aux = [labeled_class, []])
        # self.train_generator1 = torch.utils.data.DataLoader(self.train_data1, self.batch_size1, num_workers = 5, shuffle = True, pin_memory=True, drop_last=True)

        self.train_data2 = data_class(cfg['path_info']['data_home2'], train = True, label = labeled, aux = [labeled_class, unlabeled_class])
        self.train_generator2 = torch.utils.data.DataLoader(self.train_data2, self.batch_size2, num_workers = 5, shuffle = True, pin_memory=True, drop_last=True)

        self.test_data = data_class(cfg['path_info']['data_home2'], train = False, label = labeled, aux = [labeled_class, unlabeled_class + test_class])
        self.test_generator = torch.utils.data.DataLoader(self.test_data, batch_size, num_workers = 5, shuffle = True, pin_memory=True, drop_last=False)

        self.enc2 = nn.Identity()
        # self.enc_c = nn.Identity()
        self.dec_c = nn.Identity()
        try:
            w = float(cfg['train_info']['classification_weight'])
            self.bce = nn.BCEWithLogitsLoss(weight = torch.cat((torch.ones(self.y_dim - 1)/(self.y_dim - 1)*(1-w),torch.Tensor([w]))).to(self.device))
        except:
            self.bce = nn.BCEWithLogitsLoss()

    def generate_prior(self, n):
        return self.z_sampler(n, self.yz_dim + self.z_dim, device = self.device)

    def encode_s(self, x):
        xx = self.embed_data(x)
        return torch.cat((self.enc2(xx), self.enc(xx)), dim = 1)

    # def encode(self, x, y):
    #     return torch.cat((self.enc_c(y[:,:-1]), self.enc(self.embed_data(x))), dim = 1)
    
    def decode(self, z):
        return self.dec(z)

    def decode_c(self, z):
        return self.dec_c(z[:,0:self.yz_dim])
        
    def forward(self, x, y):
        return self.decode(self.encode(x, y))

    def classify_loss(self, recon, y):
        return self.bce(recon, y)

    def train(self, resume = False):
        train_main_list = []
        train_main2_list = []
        train_penalty_list = []
        test_main_list = []
        test_main2_list = []
        test_penalty_list = []

        for net in self.encoder_trainable:
            net.train()
        for net in self.decoder_trainable:
            net.train()
            
        if self.encoder_pretrain:
            self.pretrain_encoder()
            self.log.info('Pretraining Ended!')
            
        if len(self.tensorboard_dir) > 0:
            self.writer = SummaryWriter(self.tensorboard_dir)
            
        optimizer_main = optim.Adam(sum([list(net.parameters()) for net in self.encoder_trainable], []) + sum([list(net.parameters()) for net in self.decoder_trainable], []), lr = self.lr, betas = (self.beta1, 0.999))
        optimizer_adv = optim.Adam(sum([list(net.parameters()) for net in self.discriminator_trainable], []), lr = self.lr_adv, betas = (self.beta1_adv, 0.999))

        start_epoch = 0
        scheduler_main = self.lr_scheduler(optimizer_main)
        scheduler_adv = self.lr_scheduler(optimizer_adv)

        if resume:
            checkpoint = torch.load(self.save_state)
            start_epoch = checkpoint['epoch']
            self.load_state_dict(checkpoint['model_state_dict'])
            optimizer_main.load_state_dict(checkpoint['optimizer_main_state_dict'])
            optimizer_adv.load_state_dict(checkpoint['optimizer_adv_state_dict'])
            if len(self.lr_schedule) > 0:
                scheduler_main.load_state_dict(checkpoint['scheduler_main'])
                scheduler_adv.load_state_dict(checkpoint['scheduler_adv'])

        trep = len(self.train_generator2)
        ftrep = len(str(trep))
        ttep = len(self.test_generator)
        fttep = len(str(ttep))
        fep = len(str(self.num_epoch))
        
        self.log.info('------------------------------------------------------------')
        self.log.info('Training Start!')
        start_time = time.time()
        
        for epoch in range(start_epoch, self.num_epoch):
            # train_step
            train_loss_main = inc_avg()
            train_loss_main2 = inc_avg()
            train_loss_penalty = inc_avg()

            for i, (data, condition) in enumerate(self.train_generator2):
                # with label
                n = len(data)
                prior_z = self.generate_prior(n)

                x = data.to(self.device)
                y = condition.to(self.device)
                fake_latent = self.encode_s(x)

                if self.lamb > 0:
                    for net in self.discriminator_trainable:
                        net.zero_grad()
                    adv = self.adv_loss(prior_z, fake_latent)
                    obj_adv = self.lamb * adv
                    obj_adv.backward()
                    optimizer_adv.step()

                for net in self.encoder_trainable:
                    net.zero_grad()
                for net in self.decoder_trainable:
                    net.zero_grad()

                # without label
                fake_latent = self.encode_s(x)
                recon = self.decode(fake_latent)
                recon_y = self.decode_c(fake_latent)

                loss1 = self.main_loss(x, recon)
                loss2 = self.classify_loss(recon_y, y)

                loss = loss1 + self.lamb2 * loss2

                if self.lamb > 0:
                    penalty = self.penalty_loss(fake_latent)
                    obj = loss + self.lamb * penalty
                else:
                    obj = loss

                obj.backward()
                optimizer_main.step()
                
                train_loss_main.append(loss1.item(), n)
                train_loss_main2.append(loss2.item(), n)
                if self.lamb > 0:
                    train_loss_penalty.append(penalty.item(), n)
                
                print(f'[{i+1:0{ftrep}}/{trep}]  train_main: {train_loss_main.avg:.4f}  train_main2: {train_loss_main2.avg:.4f}  train_penalty: {train_loss_penalty.avg:.4f}', end = "\r")

            train_main_list.append(train_loss_main.avg)
            train_main2_list.append(train_loss_main2.avg)
            train_penalty_list.append(train_loss_penalty.avg)

            if len(self.tensorboard_dir) > 0:
                self.writer.add_scalar('train/main', train_loss_main.avg, epoch)
                self.writer.add_scalar('train/penalty', train_loss_penalty.avg, epoch)
                if self.cfg['train_info'].getboolean('histogram'):
                    for param_tensor in self.state_dict():
                        self.writer.add_histogram(param_tensor, self.state_dict()[param_tensor].detach().to('cpu').numpy().flatten(), epoch)
            
            # validation_step
            test_loss_main = inc_avg()
            test_loss_main2 = inc_avg()
            test_loss_penalty = inc_avg()

            if self.validate_batch:
                for i, (data, condition) in enumerate(self.test_generator):
                    n = len(data)
                    # test without label
                    x = data.to(self.device)
                    y = condition.to(self.device)
                    fake_latent = self.encode_s(x)
                    recon = self.decode(fake_latent)
                    recon_y = self.decode_c(fake_latent)

                    loss1 = self.main_loss(x, recon)
                    loss2 = self.classify_loss(recon_y, y)

                    test_loss_main.append(loss1.item(), n)
                    test_loss_main2.append(loss2.item(), n)
                    if self.lamb > 0:
                        test_loss_penalty.append(self.penalty_loss(fake_latent).item(), n)
                    print(f'[{i+1:0{fttep}}/{ttep}]  test_main: {test_loss_main.avg:.4f}  test_main2: {test_loss_main2.avg:.4f}  test_penalty: {test_loss_penalty.avg:.4f}', end = "\r")

                test_main_list.append(test_loss_main.avg)
                test_main2_list.append(test_loss_main2.avg)
                test_penalty_list.append(test_loss_penalty.avg)
                self.log.info(f'[{epoch + 1:0{fep}}/{self.num_epoch}]  train_main: {train_loss_main.avg:.6e}  train_penalty: {train_loss_penalty.avg:.6e}  train_main2: {train_loss_main2.avg:.4f}  train_penalty: {train_loss_penalty.avg:.6e}  test_main: {test_loss_main.avg:.6e}  test_main2: {test_loss_main2.avg:.6e}  test_penalty: {test_loss_penalty.avg:.6e}')

                # Additional test set
                data, condition = next(iter(self.test_generator))

                n = len(data)
                prior_z = self.generate_prior(n)
                x = data.to(self.device)
                # y = condition.to(self.device)
                fake_latent = self.encode_s(x)
                recon = self.decode(fake_latent)

                if len(self.tensorboard_dir) > 0:
                    self.writer.add_scalar('test/main', test_loss_main.avg, epoch)
                    self.writer.add_scalar('test/penalty', test_loss_penalty.avg, epoch)

                    if self.lamb > 0:
                        # Embedding
                        for_embed1 = fake_latent.detach().to('cpu').numpy()
                        for_embed2 = prior_z.detach().to('cpu').numpy()
                        label = ['fake']*len(for_embed1) + ['prior']*len(for_embed2)
                        self.writer.add_embedding(np.concatenate((for_embed1, for_embed2)), metadata = label, global_step = epoch)

                        # Sample Generation
                        test_dec = self.decode(prior_z).detach().to('cpu').numpy()
                        self.writer.add_images('generated_sample', (test_dec[0:32])*0.5 + 0.5, epoch)

                    # Reconstruction
                    self.writer.add_images('reconstruction', (np.concatenate((x.detach().to('cpu').numpy()[0:16], recon.detach().to('cpu').numpy()[0:16])))*0.5 + 0.5, epoch)
                    self.writer.flush()

                if len(self.save_img_path) > 0:
                    save_sample_images('%s/recon' % self.save_img_path, epoch, (np.concatenate((x.detach().to('cpu').numpy()[0:32], recon.detach().to('cpu').numpy()[0:32])))*0.5 + 0.5)
                    plt.close()
                    if self.lamb > 0:
                        # Sample Generation
                        test_dec = self.decode(prior_z).detach().to('cpu').numpy()
                        save_sample_images('%s/gen' % self.save_img_path, epoch, (test_dec[0:64])*0.5 + 0.5)
                        plt.close()
                
                if self.save_best:
                    obj = test_loss_main.avg + self.lamb * test_loss_penalty.avg
                    if self.best_obj[1] > obj:
                        self.best_obj[0] = epoch + 1
                        self.best_obj[1] = obj
                        self.save(self.save_path)
                        self.log.info("model saved, obj: %.6e" % obj)
                else:
                    self.save(self.save_path)
                    # self.log.info("model saved at: %s" % self.save_path)
                
            scheduler_main.step()
            if self.lamb > 0:
                scheduler_adv.step()

            if len(self.save_state) > 0:
                save_dict = {
                    'epoch':epoch + 1,
                    'model_state_dict':self.state_dict(),
                    'optimizer_main_state_dict':optimizer_main.state_dict(),
                    'optimizer_adv_state_dict':optimizer_adv.state_dict()
                } 
                if len(self.lr_schedule) > 0:
                    save_dict['scheduler_main'] = scheduler_main.state_dict()
                    save_dict['scheduler_adv'] = scheduler_adv.state_dict()
                torch.save(save_dict, self.save_state)
            
        if not self.validate_batch:
            self.save(self.save_path)

        self.log.info('Training Finished!')
        self.log.info("Elapsed time: %.3fs" % (time.time() - start_time))

        if len(self.tensorboard_dir) > 0:
            self.writer.close()

        self.hist = np.array([train_main_list, train_main2_list, train_penalty_list, test_main_list, test_main2_list, test_penalty_list])

class SSWAE_HSIC_abstract(SSWAE_GAN_abstract):
    def __init__(self, cfg, log, device = 'cpu', verbose = 1):
        super(SSWAE_HSIC_abstract, self).__init__(cfg, log, device, verbose)
        self.lamb2 = float(cfg['train_info']['lambda2'])
        self.lamb3 = float(cfg['train_info']['lambda_mmd'])
        self.lamb4 = float(cfg['train_info']['lambda_hsic'])

    def k(self, x, y, diag = True):
        stat = 0.
        for scale in [.1, .2, .5, 1., 2., 5., 10.]:
            C = scale*2*self.z_dim*2
            kernel = (C/(C + (x.unsqueeze(0) - y.unsqueeze(1)).pow(2).sum(dim = 2)))
            if diag:
                stat += kernel.sum()
            else:
                stat += kernel.sum() - kernel.diag().sum()
        return stat

    """
    Refers to original Tensorflow implementation: https://github.com/romain-lopez/HCV
    Refers to original implementations
        - https://github.com/kacperChwialkowski/HSIC
        - https://cran.r-project.org/web/packages/dHSIC/index.html
    """
    def bandwidth(self, d):
        gz = 2 * gamma(0.5 * (d+1)) / gamma(0.5 * d)
        return 1. / (2. * gz**2)

    def knl(self, x, y, gam=1.):
        dist_table = (x.unsqueeze(0) - y.unsqueeze(1)).pow(2).sum(dim = 2)
        return (-gam * dist_table).exp().transpose(0,1)

    def hsic(self, x, y):
        dx = x.shape[1]
        dy = y.shape[1]

        xx = self.knl(x, x, gam=self.bandwidth(dx))
        yy = self.knl(y, y, gam=self.bandwidth(dy))

        res = ((xx*yy).mean()) + (xx.mean()) * (yy.mean())
        res -= 2*((xx.mean(dim=1))*(yy.mean(dim=1))).mean()
        return res.clamp(min = 1e-16).sqrt()

    def penalty_loss(self, q, prior_z, n):
        x = q[:,0:self.yz_dim]
        y = prior_z[:,0:self.yz_dim]
        mmd = (self.k(x,x, False) + self.k(y,y, False))/(n*(n-1)) - 2*self.k(x,y, True)/(n*n)

        z = q[:,self.yz_dim:]
        qz = self.discriminate(z)
        gan = self.disc_loss(qz, torch.ones_like(qz))

        hsic = self.hsic(x, z)

        return gan, mmd, hsic

    def train(self, resume = False):
        train_main_list = []
        train_main2_list = []
        train_penalty_list = []
        train_penalty2_list = []
        train_penalty3_list = []
        test_main_list = []
        test_main2_list = []
        test_penalty_list = []
        test_penalty2_list = []
        test_penalty3_list = []

        for net in self.encoder_trainable:
            net.train()
        for net in self.decoder_trainable:
            net.train()
            
        if self.encoder_pretrain:
            self.pretrain_encoder()
            self.log.info('Pretraining Ended!')
            
        if len(self.tensorboard_dir) > 0:
            self.writer = SummaryWriter(self.tensorboard_dir)
            
        optimizer_main = optim.Adam(sum([list(net.parameters()) for net in self.encoder_trainable], []) + sum([list(net.parameters()) for net in self.decoder_trainable], []), lr = self.lr, betas = (self.beta1, 0.999))
        optimizer_adv = optim.Adam(sum([list(net.parameters()) for net in self.discriminator_trainable], []), lr = self.lr_adv, betas = (self.beta1_adv, 0.999))

        start_epoch = 0
        scheduler_main = self.lr_scheduler(optimizer_main)
        scheduler_adv = self.lr_scheduler(optimizer_adv)

        if resume:
            checkpoint = torch.load(self.save_state)
            start_epoch = checkpoint['epoch']
            self.load_state_dict(checkpoint['model_state_dict'])
            optimizer_main.load_state_dict(checkpoint['optimizer_main_state_dict'])
            optimizer_adv.load_state_dict(checkpoint['optimizer_adv_state_dict'])
            if len(self.lr_schedule) > 0:
                scheduler_main.load_state_dict(checkpoint['scheduler_main'])
                scheduler_adv.load_state_dict(checkpoint['scheduler_adv'])

        iter_per_epoch = min(len(self.train_generator1), len(self.train_generator2))
        n1 = self.batch_size1
        n2 = self.batch_size2

        trep = iter_per_epoch
        ftrep = len(str(iter_per_epoch))
        ttep = len(self.test_generator)
        fttep = len(str(ttep))
        fep = len(str(self.num_epoch))
        
        self.log.info('------------------------------------------------------------')
        self.log.info('Training Start!')
        start_time = time.time()
        
        for epoch in range(start_epoch, self.num_epoch):
            # train_step
            train_loss_main = inc_avg()
            train_loss_main2 = inc_avg()
            train_loss_penalty = inc_avg()
            train_loss_penalty2 = inc_avg()
            train_loss_penalty3 = inc_avg()

            tr1 = iter(self.train_generator1)
            tr2 = iter(self.train_generator2)

            for i in range(iter_per_epoch):
                # with label
                data1, condition1 = next(tr1)
                data2, condition2 = next(tr2)
                prior_z = self.generate_prior(n1+n2)

                if self.lamb > 0:
                    for net in self.discriminator_trainable:
                        net.zero_grad()

                    x1 = data1.to(self.device)
                    y1 = condition1.to(self.device)
                    fake_latent1 = self.encode(x1, y1)[:,self.yz_dim:]

                    x2 = data2.to(self.device)
                    fake_latent2 = self.encode_s(x2)[:,self.yz_dim:]

                    fake_latent = torch.cat((fake_latent1, fake_latent2), dim = 0)

                    adv = self.adv_loss(prior_z[:,self.yz_dim:], fake_latent)
                    obj_adv = self.lamb * adv
                    obj_adv.backward()
                    optimizer_adv.step()

                for net in self.encoder_trainable:
                    net.zero_grad()
                for net in self.decoder_trainable:
                    net.zero_grad()

                # with label

                x1 = data1.to(self.device)
                y1 = condition1.to(self.device)
                fake_latent1 = self.encode(x1, y1)
                recon = self.decode(fake_latent1)
                recon_y = self.decode_c(fake_latent1)
                
                loss1 = self.main_loss(x1, recon)
                loss2 = self.classify_loss(recon_y, y1)
                loss = loss1 + self.lamb2 * loss2

                # without label

                x2 = data2.to(self.device)
                y2 = condition2.to(self.device)
                fake_latent2 = self.encode_s(x2)
                recon2 = self.decode(fake_latent2)
                recon_y2 = self.decode_c(fake_latent2)

                los1 = self.main_loss(x2, recon2)
                los2 = self.classify_loss(recon_y2, y2)
                los = los1 + self.lamb2 * los2

                fake_latent = torch.cat((fake_latent1, fake_latent2), dim = 0)

                if self.lamb > 0:
                    gan, mmd, hsic = self.penalty_loss(fake_latent, prior_z, n1+n2)
                    obj = loss *(n1/(n1+n2))  + los *(n2/(n1+n2)) + self.lamb * gan + self.lamb3 * mmd + self.lamb4 * hsic
                else:
                    obj = loss *(n1/(n1+n2))  + los *(n2/(n1+n2))

                obj.backward()
                optimizer_main.step()
                
                train_loss_main.append((loss1 *(n1/(n1+n2)) + los1 *(n2/(n1+n2))).item(), n1+n2)
                train_loss_main2.append((loss2 *(n1/(n1+n2)) + los2 *(n2/(n1+n2))).item(), n1+n2)
                if self.lamb > 0:
                    train_loss_penalty.append(gan.item(), n1+n2)
                    train_loss_penalty2.append(mmd.item(), n1+n2)
                    train_loss_penalty3.append(hsic.item(), n1+n2)
                
                print(f"[{i+1:0{ftrep}}/{trep}]  train_main: {train_loss_main.avg:.4f}  train_main2: {train_loss_main2.avg:.4f}  train_penalty: {train_loss_penalty.avg:.4f} ", 
                f"train_penalty2: {train_loss_penalty2.avg:.4f}  train_penalty3: {train_loss_penalty3.avg:.4f}",
                end = '\r')

            train_main_list.append(train_loss_main.avg)
            train_main2_list.append(train_loss_main2.avg)
            train_penalty_list.append(train_loss_penalty.avg)
            train_penalty2_list.append(train_loss_penalty2.avg)
            train_penalty3_list.append(train_loss_penalty3.avg)

            if len(self.tensorboard_dir) > 0:
                self.writer.add_scalar('train/main', train_loss_main.avg, epoch)
                self.writer.add_scalar('train/penalty', train_loss_penalty.avg, epoch)
                if self.cfg['train_info'].getboolean('histogram'):
                    for param_tensor in self.state_dict():
                        self.writer.add_histogram(param_tensor, self.state_dict()[param_tensor].detach().to('cpu').numpy().flatten(), epoch)
            
            # validation_step
            test_loss_main = inc_avg()
            test_loss_main2 = inc_avg()
            test_loss_penalty = inc_avg()
            test_loss_penalty2 = inc_avg()
            test_loss_penalty3 = inc_avg()

            if self.validate_batch:
                for i, (data, condition) in enumerate(self.test_generator):
                    n = len(data)
                    prior_z = self.generate_prior(n)
                    # test without label
                    x = data.to(self.device)
                    y = condition.to(self.device)
                    fake_latent = self.encode_s(x)
                    recon = self.decode(fake_latent)
                    recon_y = self.decode_c(fake_latent)

                    loss1 = self.main_loss(x, recon)
                    loss2 = self.classify_loss(recon_y, y)

                    test_loss_main.append(loss1.item(), n)
                    test_loss_main2.append(loss2.item(), n)
                    if self.lamb > 0:
                        gan, mmd, hsic = self.penalty_loss(fake_latent, prior_z, n)
                        test_loss_penalty.append(gan.item(), n)
                        test_loss_penalty2.append(mmd.item(), n)
                        test_loss_penalty3.append(hsic.item(), n)
                    print(f"[{i+1:0{fttep}}/{ttep}]  test_main: {test_loss_main.avg:.4f}  test_main2: {test_loss_main2.avg:.4f}  test_penalty: {test_loss_penalty.avg:.4f} ",
                    f"test_penalty2: {test_loss_penalty2.avg:.4f}  test_penalty3: {test_loss_penalty3.avg:.4f}", end = "\r")

                test_main_list.append(test_loss_main.avg)
                test_main2_list.append(test_loss_main2.avg)
                test_penalty_list.append(test_loss_penalty.avg)
                test_penalty2_list.append(test_loss_penalty2.avg)
                test_penalty3_list.append(test_loss_penalty3.avg)

                pp = f"[{epoch + 1:0{fep}}/{self.num_epoch}]  train_main: {train_loss_main.avg:.6e}  train_penalty: {train_loss_penalty.avg:.6e}  train_main2: {train_loss_main2.avg:.4f}  "
                pp += f"train_penalty: {train_loss_penalty.avg:.6e}  train_penalty: {train_loss_penalty2.avg:.6e}  train_penalty: {train_loss_penalty3.avg:.6e}\ntest_main: {test_loss_main.avg:.6e}  "
                pp += f"test_main2: {test_loss_main2.avg:.6e}  test_penalty: {test_loss_penalty.avg:.6e}  test_penalty: {test_loss_penalty2.avg:.6e}  test_penalty: {test_loss_penalty3.avg:.6e}"
                self.log.info(pp)

                # Additional test set
                data, condition = next(iter(self.test_generator))

                n = len(data)
                prior_z = self.generate_prior(n)
                x = data.to(self.device)
                # y = condition.to(self.device)
                fake_latent = self.encode_s(x)
                recon = self.decode(fake_latent)

                if len(self.tensorboard_dir) > 0:
                    self.writer.add_scalar('test/main', test_loss_main.avg, epoch)
                    self.writer.add_scalar('test/penalty', test_loss_penalty.avg, epoch)

                    if self.lamb > 0:
                        # Embedding
                        for_embed1 = fake_latent.detach().to('cpu').numpy()
                        for_embed2 = prior_z.detach().to('cpu').numpy()
                        label = ['fake']*len(for_embed1) + ['prior']*len(for_embed2)
                        self.writer.add_embedding(np.concatenate((for_embed1, for_embed2)), metadata = label, global_step = epoch)

                        # Sample Generation
                        test_dec = self.decode(prior_z).detach().to('cpu').numpy()
                        self.writer.add_images('generated_sample', (test_dec[0:32])*0.5 + 0.5, epoch)

                    # Reconstruction
                    self.writer.add_images('reconstruction', (np.concatenate((x.detach().to('cpu').numpy()[0:16], recon.detach().to('cpu').numpy()[0:16])))*0.5 + 0.5, epoch)
                    self.writer.flush()

                if len(self.save_img_path) > 0:
                    save_sample_images('%s/recon' % self.save_img_path, epoch, (np.concatenate((x.detach().to('cpu').numpy()[0:32], recon.detach().to('cpu').numpy()[0:32])))*0.5 + 0.5)
                    plt.close()
                    if self.lamb > 0:
                        # Sample Generation
                        test_dec = self.decode(prior_z).detach().to('cpu').numpy()
                        save_sample_images('%s/gen' % self.save_img_path, epoch, (test_dec[0:64])*0.5 + 0.5)
                        plt.close()
                
                if self.save_best:
                    obj = test_loss_main.avg + self.lamb * test_loss_penalty.avg
                    if self.best_obj[1] > obj:
                        self.best_obj[0] = epoch + 1
                        self.best_obj[1] = obj
                        self.save(self.save_path)
                        self.log.info("model saved, obj: %.6e" % obj)
                else:
                    self.save(self.save_path)
                    # self.log.info("model saved at: %s" % self.save_path)
                
            scheduler_main.step()
            if self.lamb > 0:
                scheduler_adv.step()

            if len(self.save_state) > 0:
                save_dict = {
                    'epoch':epoch + 1,
                    'model_state_dict':self.state_dict(),
                    'optimizer_main_state_dict':optimizer_main.state_dict(),
                    'optimizer_adv_state_dict':optimizer_adv.state_dict()
                } 
                if len(self.lr_schedule) > 0:
                    save_dict['scheduler_main'] = scheduler_main.state_dict()
                    save_dict['scheduler_adv'] = scheduler_adv.state_dict()
                torch.save(save_dict, self.save_state)
            
        if not self.validate_batch:
            self.save(self.save_path)

        self.log.info('Training Finished!')
        self.log.info("Elapsed time: %.3fs" % (time.time() - start_time))

        if len(self.tensorboard_dir) > 0:
            self.writer.close()

        self.hist = np.array([train_main_list, train_main2_list, train_penalty_list, train_penalty2_list, train_penalty3_list, 
        test_main_list, test_main2_list, test_penalty_list, test_penalty2_list, test_penalty3_list])

class SSWAE_HSIC_dev(SSWAE_GAN_dev):
    def __init__(self, cfg, log, device = 'cpu', verbose = 1):
        super(SSWAE_HSIC_dev, self).__init__(cfg, log, device, verbose)
        self.lamb2 = float(cfg['train_info']['lambda2'])
        self.lamb3 = float(cfg['train_info']['lambda_mmd'])
        self.lamb4 = float(cfg['train_info']['lambda_hsic'])

    def k(self, x, y, diag = True):
        stat = 0.
        for scale in [.1, .2, .5, 1., 2., 5., 10.]:
            C = scale*2*self.z_dim*2
            kernel = (C/(C + (x.unsqueeze(0) - y.unsqueeze(1)).pow(2).sum(dim = 2)))
            if diag:
                stat += kernel.sum()
            else:
                stat += kernel.sum() - kernel.diag().sum()
        return stat

    """
    Refers to original Tensorflow implementation: https://github.com/romain-lopez/HCV
    Refers to original implementations
        - https://github.com/kacperChwialkowski/HSIC
        - https://cran.r-project.org/web/packages/dHSIC/index.html
    """
    def bandwidth(self, d):
        gz = 2 * gamma(0.5 * (d+1)) / gamma(0.5 * d)
        return 1. / (2. * gz**2)

    def knl(self, x, y, gam=1.):
        dist_table = (x.unsqueeze(0) - y.unsqueeze(1)).pow(2).sum(dim = 2)
        return (-gam * dist_table).exp().transpose(0,1)

    def hsic(self, x, y):
        dx = x.shape[1]
        dy = y.shape[1]

        xx = self.knl(x, x, gam=self.bandwidth(dx))
        yy = self.knl(y, y, gam=self.bandwidth(dy))

        res = ((xx*yy).mean()) + (xx.mean()) * (yy.mean())
        res -= 2*((xx.mean(dim=1))*(yy.mean(dim=1))).mean()
        return res.clamp(min = 1e-16).sqrt()

    def penalty_loss(self, q, prior_z, n):
        x = q[:,0:self.yz_dim]
        y = prior_z[:,0:self.yz_dim]
        mmd = (self.k(x,x, False) + self.k(y,y, False))/(n*(n-1)) - 2*self.k(x,y, True)/(n*n)

        z = q[:,self.yz_dim:]
        qz = self.discriminate(z)
        gan = self.disc_loss(qz, torch.ones_like(qz))

        hsic = self.hsic(x, z)

        return gan, mmd, hsic

    def train(self, resume = False):
        train_main_list = []
        train_main2_list = []
        train_penalty_list = []
        train_penalty2_list = []
        train_penalty3_list = []
        test_main_list = []
        test_main2_list = []
        test_penalty_list = []
        test_penalty2_list = []
        test_penalty3_list = []

        for net in self.encoder_trainable:
            net.train()
        for net in self.decoder_trainable:
            net.train()
            
        if self.encoder_pretrain:
            self.pretrain_encoder()
            self.log.info('Pretraining Ended!')
            
        if len(self.tensorboard_dir) > 0:
            self.writer = SummaryWriter(self.tensorboard_dir)
            
        optimizer_main = optim.Adam(sum([list(net.parameters()) for net in self.encoder_trainable], []) + sum([list(net.parameters()) for net in self.decoder_trainable], []), lr = self.lr, betas = (self.beta1, 0.999))
        optimizer_adv = optim.Adam(sum([list(net.parameters()) for net in self.discriminator_trainable], []), lr = self.lr_adv, betas = (self.beta1_adv, 0.999))

        start_epoch = 0
        scheduler_main = self.lr_scheduler(optimizer_main)
        scheduler_adv = self.lr_scheduler(optimizer_adv)

        if resume:
            checkpoint = torch.load(self.save_state)
            start_epoch = checkpoint['epoch']
            self.load_state_dict(checkpoint['model_state_dict'])
            optimizer_main.load_state_dict(checkpoint['optimizer_main_state_dict'])
            optimizer_adv.load_state_dict(checkpoint['optimizer_adv_state_dict'])
            if len(self.lr_schedule) > 0:
                scheduler_main.load_state_dict(checkpoint['scheduler_main'])
                scheduler_adv.load_state_dict(checkpoint['scheduler_adv'])

        trep = len(self.train_generator2)
        ftrep = len(str(trep))
        ttep = len(self.test_generator)
        fttep = len(str(ttep))
        fep = len(str(self.num_epoch))
        
        self.log.info('------------------------------------------------------------')
        self.log.info('Training Start!')
        start_time = time.time()
        
        for epoch in range(start_epoch, self.num_epoch):
            # train_step
            train_loss_main = inc_avg()
            train_loss_main2 = inc_avg()
            train_loss_penalty = inc_avg()
            train_loss_penalty2 = inc_avg()
            train_loss_penalty3 = inc_avg()

            for i, (data,condition) in enumerate(self.train_generator2):
                # with label
                n = len(data)
                prior_z = self.generate_prior(n)

                x = data.to(self.device)
                y = condition.to(self.device)
                fake_latent = self.encode_s(x)

                if self.lamb > 0:
                    for net in self.discriminator_trainable:
                        net.zero_grad()
                    adv = self.adv_loss(prior_z[:,self.yz_dim:], fake_latent[:,self.yz_dim:])
                    obj_adv = self.lamb * adv
                    obj_adv.backward()
                    optimizer_adv.step()

                for net in self.encoder_trainable:
                    net.zero_grad()
                for net in self.decoder_trainable:
                    net.zero_grad()

                # without label
                fake_latent = self.encode_s(x)
                recon = self.decode(fake_latent)
                recon_y = self.decode_c(fake_latent)

                loss1 = self.main_loss(x, recon)
                loss2 = self.classify_loss(recon_y, y)
                loss = loss1 + self.lamb2 * loss2

                if self.lamb > 0:
                    gan, mmd, hsic = self.penalty_loss(fake_latent, prior_z, n)
                    obj = loss + self.lamb * gan + self.lamb3 * mmd + self.lamb4 * hsic
                else:
                    obj = loss

                obj.backward()
                optimizer_main.step()
                
                train_loss_main.append(loss1.item(), n)
                train_loss_main2.append(loss2.item(), n)
                if self.lamb > 0:
                    train_loss_penalty.append(gan.item(), n)
                    train_loss_penalty2.append(mmd.item(), n)
                    train_loss_penalty3.append(hsic.item(), n)
                
                print(f"[{i+1:0{ftrep}}/{trep}]  train_main: {train_loss_main.avg:.4f}  train_main2: {train_loss_main2.avg:.4f}  train_penalty: {train_loss_penalty.avg:.4f} ", 
                f"train_penalty2: {train_loss_penalty2.avg:.4f}  train_penalty3: {train_loss_penalty3.avg:.4f}",
                end = '\r')

            train_main_list.append(train_loss_main.avg)
            train_main2_list.append(train_loss_main2.avg)
            train_penalty_list.append(train_loss_penalty.avg)
            train_penalty2_list.append(train_loss_penalty2.avg)
            train_penalty3_list.append(train_loss_penalty3.avg)

            if len(self.tensorboard_dir) > 0:
                self.writer.add_scalar('train/main', train_loss_main.avg, epoch)
                self.writer.add_scalar('train/penalty', train_loss_penalty.avg, epoch)
                if self.cfg['train_info'].getboolean('histogram'):
                    for param_tensor in self.state_dict():
                        self.writer.add_histogram(param_tensor, self.state_dict()[param_tensor].detach().to('cpu').numpy().flatten(), epoch)
            
            # validation_step
            test_loss_main = inc_avg()
            test_loss_main2 = inc_avg()
            test_loss_penalty = inc_avg()
            test_loss_penalty2 = inc_avg()
            test_loss_penalty3 = inc_avg()

            if self.validate_batch:
                for i, (data, condition) in enumerate(self.test_generator):
                    n = len(data)
                    prior_z = self.generate_prior(n)
                    # test without label
                    x = data.to(self.device)
                    y = condition.to(self.device)
                    fake_latent = self.encode_s(x)
                    recon = self.decode(fake_latent)
                    recon_y = self.decode_c(fake_latent)

                    loss1 = self.main_loss(x, recon)
                    loss2 = self.classify_loss(recon_y, y)

                    test_loss_main.append(loss1.item(), n)
                    test_loss_main2.append(loss2.item(), n)
                    if self.lamb > 0:
                        gan, mmd, hsic = self.penalty_loss(fake_latent, prior_z, n)
                        test_loss_penalty.append(gan.item(), n)
                        test_loss_penalty2.append(mmd.item(), n)
                        test_loss_penalty3.append(hsic.item(), n)
                    print(f"[{i+1:0{fttep}}/{ttep}]  test_main: {test_loss_main.avg:.4f}  test_main2: {test_loss_main2.avg:.4f}  test_penalty: {test_loss_penalty.avg:.4f} ",
                    f"test_penalty2: {test_loss_penalty2.avg:.4f}  test_penalty3: {test_loss_penalty3.avg:.4f}", end = "\r")

                test_main_list.append(test_loss_main.avg)
                test_main2_list.append(test_loss_main2.avg)
                test_penalty_list.append(test_loss_penalty.avg)
                test_penalty2_list.append(test_loss_penalty2.avg)
                test_penalty3_list.append(test_loss_penalty3.avg)

                pp = f"[{epoch + 1:0{fep}}/{self.num_epoch}]  train_main: {train_loss_main.avg:.6e}  train_penalty: {train_loss_penalty.avg:.6e}  train_main2: {train_loss_main2.avg:.4f}  "
                pp += f"train_penalty: {train_loss_penalty.avg:.6e}  train_penalty: {train_loss_penalty2.avg:.6e}  train_penalty: {train_loss_penalty3.avg:.6e}\ntest_main: {test_loss_main.avg:.6e}  "
                pp += f"test_main2: {test_loss_main2.avg:.6e}  test_penalty: {test_loss_penalty.avg:.6e}  test_penalty: {test_loss_penalty2.avg:.6e}  test_penalty: {test_loss_penalty3.avg:.6e}"
                self.log.info(pp)

                # Additional test set
                data, condition = next(iter(self.test_generator))

                n = len(data)
                prior_z = self.generate_prior(n)
                x = data.to(self.device)
                # y = condition.to(self.device)
                fake_latent = self.encode_s(x)
                recon = self.decode(fake_latent)

                if len(self.tensorboard_dir) > 0:
                    self.writer.add_scalar('test/main', test_loss_main.avg, epoch)
                    self.writer.add_scalar('test/penalty', test_loss_penalty.avg, epoch)

                    if self.lamb > 0:
                        # Embedding
                        for_embed1 = fake_latent.detach().to('cpu').numpy()
                        for_embed2 = prior_z.detach().to('cpu').numpy()
                        label = ['fake']*len(for_embed1) + ['prior']*len(for_embed2)
                        self.writer.add_embedding(np.concatenate((for_embed1, for_embed2)), metadata = label, global_step = epoch)

                        # Sample Generation
                        test_dec = self.decode(prior_z).detach().to('cpu').numpy()
                        self.writer.add_images('generated_sample', (test_dec[0:32])*0.5 + 0.5, epoch)

                    # Reconstruction
                    self.writer.add_images('reconstruction', (np.concatenate((x.detach().to('cpu').numpy()[0:16], recon.detach().to('cpu').numpy()[0:16])))*0.5 + 0.5, epoch)
                    self.writer.flush()

                if len(self.save_img_path) > 0:
                    save_sample_images('%s/recon' % self.save_img_path, epoch, (np.concatenate((x.detach().to('cpu').numpy()[0:32], recon.detach().to('cpu').numpy()[0:32])))*0.5 + 0.5)
                    plt.close()
                    if self.lamb > 0:
                        # Sample Generation
                        test_dec = self.decode(prior_z).detach().to('cpu').numpy()
                        save_sample_images('%s/gen' % self.save_img_path, epoch, (test_dec[0:64])*0.5 + 0.5)
                        plt.close()
                
                if self.save_best:
                    obj = test_loss_main.avg + self.lamb * test_loss_penalty.avg
                    if self.best_obj[1] > obj:
                        self.best_obj[0] = epoch + 1
                        self.best_obj[1] = obj
                        self.save(self.save_path)
                        self.log.info("model saved, obj: %.6e" % obj)
                else:
                    self.save(self.save_path)
                    # self.log.info("model saved at: %s" % self.save_path)
                
            scheduler_main.step()
            if self.lamb > 0:
                scheduler_adv.step()

            if len(self.save_state) > 0:
                save_dict = {
                    'epoch':epoch + 1,
                    'model_state_dict':self.state_dict(),
                    'optimizer_main_state_dict':optimizer_main.state_dict(),
                    'optimizer_adv_state_dict':optimizer_adv.state_dict()
                } 
                if len(self.lr_schedule) > 0:
                    save_dict['scheduler_main'] = scheduler_main.state_dict()
                    save_dict['scheduler_adv'] = scheduler_adv.state_dict()
                torch.save(save_dict, self.save_state)
            
        if not self.validate_batch:
            self.save(self.save_path)

        self.log.info('Training Finished!')
        self.log.info("Elapsed time: %.3fs" % (time.time() - start_time))

        if len(self.tensorboard_dir) > 0:
            self.writer.close()

        self.hist = np.array([train_main_list, train_main2_list, train_penalty_list, train_penalty2_list, train_penalty3_list, 
        test_main_list, test_main2_list, test_penalty_list, test_penalty2_list, test_penalty3_list])

# class SSWAE_MMD_dev(CWAE_MMD_abstract):
#     def __init__(self, cfg, log, device = 'cpu', verbose = 1):
#         super(SSWAE_MMD_dev, self).__init__(cfg, log, device, verbose)
#         self.yz_dim = int(cfg['train_info']['yz_dim'])
#         self.lamb2 = float(cfg['train_info']['lambda2'])

#         data_class = getattr(dataset, cfg['train_info']['train_data'])
#         labeled = cfg['train_info'].getboolean('train_data_label')

#         labeled_class = cfg['train_info']['labeled_class'].replace(' ', '').split(',')
#         labeled_class = [int(i) for i in labeled_class]
#         unlabeled_class = cfg['train_info']['unlabeled_class'].replace(' ', '').split(',')
#         unlabeled_class = [int(i) for i in unlabeled_class]
#         test_class = cfg['train_info']['test_class'].replace(' ', '').split(',')
#         test_class = [int(i) for i in test_class]

#         self.batch_size1 = int(cfg['train_info']['batch_size1'])
#         self.batch_size2 = int(cfg['train_info']['batch_size2'])
#         batch_size = max(self.batch_size1, self.batch_size2)

#         self.train_data1 = data_class(cfg['path_info']['data_home2'], train = True, label = labeled, aux = [labeled_class, []])
#         self.train_generator1 = torch.utils.data.DataLoader(self.train_data1, self.batch_size1, num_workers = 5, shuffle = True, pin_memory=True, drop_last=True)

#         self.train_data2 = data_class(cfg['path_info']['data_home2'], train = True, label = labeled, aux = [labeled_class, unlabeled_class])
#         self.train_generator2 = torch.utils.data.DataLoader(self.train_data2, self.batch_size2, num_workers = 5, shuffle = True, pin_memory=True, drop_last=True)

#         self.test_data = data_class(cfg['path_info']['data_home2'], train = False, label = labeled, aux = [labeled_class, unlabeled_class + test_class])
#         self.test_generator = torch.utils.data.DataLoader(self.test_data, batch_size, num_workers = 5, shuffle = True, pin_memory=True, drop_last=False)

#         self.enc2 = nn.Identity()
#         self.enc_c = nn.Identity()
#         self.dec_c = nn.Identity()
#         try:
#             w = float(cfg['train_info']['classification_weight'])
#             self.bce = nn.BCEWithLogitsLoss(weight = torch.cat((torch.ones(self.y_dim - 1)/(self.y_dim - 1)*(1-w),torch.Tensor([w]))).to(self.device))
#         except:
#             self.bce = nn.BCEWithLogitsLoss()

#     def generate_prior(self, n):
#         return self.z_sampler(n, self.yz_dim + self.z_dim, device = self.device)

#     def encode_s(self, x):
#         xx = self.embed_data(x)
#         return torch.cat((self.enc2(torch.cat((self.enc_c(xx), xx), dim = 1)), self.enc(xx)), dim = 1)

#     def encode(self, x, y):
#         xx = self.embed_data(x)
#         return torch.cat((self.enc2(torch.cat((y, xx), dim = 1)), self.enc(xx)), dim = 1)
    
#     def decode(self, z):
#         return self.dec(z)

#     def decode_c(self, z):
#         return self.dec_c(z[:,0:self.yz_dim])
        
#     def forward(self, x, y):
#         return self.decode(self.encode(x, y))

#     def classify_loss(self, recon, y):
#         return self.bce(recon, y)

#     def k(self, x, y, diag = True):
#         stat = 0.
#         for scale in [.1, .2, .5, 1., 2., 5., 10.]:
#             C = scale*2*(self.z_dim+self.yz_dim)*2
#             kernel = (C/(C + (x.unsqueeze(0) - y.unsqueeze(1)).pow(2).sum(dim = 2)))
#             if diag:
#                 stat += kernel.sum()
#             else:
#                 stat += kernel.sum() - kernel.diag().sum()
#         return stat
        
#     def penalty_loss(self, x, y, n):
#         return (self.k(x,x, False) + self.k(y,y, False))/(n*(n-1)) - 2*self.k(x,y, True)/(n*n)

#     def train(self, resume = False):
#         self.train_main_list = []
#         self.train_main2_list = []
#         self.train_penalty_list = []
#         self.test_main_list = []
#         self.test_main2_list = []
#         self.test_penalty_list = []

#         for net in self.encoder_trainable:
#             net.train()
#         for net in self.decoder_trainable:
#             net.train()
            
#         if self.encoder_pretrain:
#             self.pretrain_encoder()
#             self.log.info('Pretraining Ended!')
            
#         if len(self.tensorboard_dir) > 0:
#             self.writer = SummaryWriter(self.tensorboard_dir)
            
#         optimizer = optim.Adam(sum([list(net.parameters()) for net in self.encoder_trainable], []) + sum([list(net.parameters()) for net in self.decoder_trainable], []), lr = self.lr, betas = (self.beta1, 0.999))

#         start_epoch = 0
#         scheduler = self.lr_scheduler(optimizer)

#         if resume:
#             checkpoint = torch.load(self.save_state)
#             start_epoch = checkpoint['epoch']
#             self.load_state_dict(checkpoint['model_state_dict'])
#             optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#             if len(self.lr_schedule) > 0:
#                 scheduler.load_state_dict(checkpoint['scheduler'])

#         self.log.info('------------------------------------------------------------')
#         self.log.info('Training Start!')
#         start_time = time.time()

#         iter_per_epoch = min(len(self.train_generator1), len(self.train_generator2))
#         n1 = self.batch_size1
#         n2 = self.batch_size2

#         for epoch in range(start_epoch, self.num_epoch):
#             # train_step
#             train_loss_main = inc_avg()
#             train_loss_main2 = inc_avg()
#             train_loss_penalty = inc_avg()

#             tr1 = iter(self.train_generator1)
#             tr2 = iter(self.train_generator2)

#             for i in range(iter_per_epoch):
#                 for net in self.encoder_trainable:
#                     net.zero_grad()
#                 for net in self.decoder_trainable:
#                     net.zero_grad()
                
#                 # with label
#                 data, condition = next(tr1)

#                 x1 = data.to(self.device)
#                 y1 = condition.to(self.device)
#                 fake_latent1 = self.encode(x1, y1)
#                 recon = self.decode(fake_latent1)
#                 recon_y = self.decode_c(fake_latent1)
                
#                 loss1 = self.main_loss(x1, recon)
#                 loss2 = self.classify_loss(recon_y, y1)
#                 loss = loss1 + self.lamb2 * loss2

#                 # without label
#                 data, condition = next(tr2)

#                 x2 = data.to(self.device)
#                 y2 = condition.to(self.device)
#                 fake_latent2 = self.encode_s(x2)
#                 recon2 = self.decode(fake_latent2)
#                 recon_y2 = self.decode_c(fake_latent2)

#                 los1 = self.main_loss(x2, recon2)
#                 los2 = self.classify_loss(recon_y2, y2)
#                 los = los1 + self.lamb2 * los2

#                 fake_latent = torch.cat((fake_latent1, fake_latent2), dim = 0)
#                 prior_z = self.generate_prior(n1 + n2)

#                 if self.lamb > 0:
#                     penalty = self.penalty_loss(fake_latent, prior_z, n1+n2)
#                     obj = loss *(n1/(n1+n2))  + los *(n2/(n1+n2)) + self.lamb * penalty
#                 else:
#                     obj = loss *(n1/(n1+n2))  + los *(n2/(n1+n2))

#                 obj.backward()
#                 optimizer.step()
                
#                 train_loss_main.append((loss1 *(n1/(n1+n2))  + los1 *(n1/(n1+n2))).item(), n1+n2)
#                 train_loss_main2.append((loss2 *(n1/(n1+n2))  + los2 *(n1/(n1+n2))).item(), n1+n2)
#                 if self.lamb > 0:
#                     train_loss_penalty.append(penalty.item(), n1+n2)
                
#                 print(f'[{i+1:0{ftrep}}/{trep}]  train_main: {train_loss_main.avg:.4f}  train_main2: {train_loss_main2.avg:.4f}  train_penalty: {train_loss_penalty.avg:.4f}', end = "\r")

#             self.train_main_list.append(train_loss_main.avg)
#             self.train_main2_list.append(train_loss_main2.avg)
#             self.train_penalty_list.append(train_loss_penalty.avg)

#             if len(self.tensorboard_dir) > 0:
#                 self.writer.add_scalar('train/main', train_loss_main.avg, epoch)
#                 self.writer.add_scalar('train/penalty', train_loss_penalty.avg, epoch)
#                 if self.cfg['train_info'].getboolean('histogram'):
#                     for param_tensor in self.state_dict():
#                         self.writer.add_histogram(param_tensor, self.state_dict()[param_tensor].detach().to('cpu').numpy().flatten(), epoch)
            
#             # validation_step
#             test_loss_main = inc_avg()
#             test_loss_main2 = inc_avg()
#             test_loss_penalty = inc_avg()

#             if self.validate_batch:
#                 for i, (data, condition) in enumerate(self.test_generator):
#                     n = len(data)

#                     # test without label
#                     prior_z = self.generate_prior(n)
#                     x = data.to(self.device)
#                     y = condition.to(self.device)
#                     fake_latent = self.encode_s(x)
#                     recon = self.decode(fake_latent)
#                     recon_y = self.decode_c(fake_latent)

#                     loss1 = self.main_loss(x, recon)
#                     loss2 = self.classify_loss(recon_y, y)

#                     test_loss_main.append(loss1.item(), n)
#                     test_loss_main2.append(loss2.item(), n)
#                     if self.lamb > 0:
#                         test_loss_penalty.append(self.penalty_loss(fake_latent, prior_z, n).item(), n)
#                     print(f'[{i+1:0{fttep}}/{ttep}]  test_main: {test_loss_main.avg:.4f}  test_main2: {test_loss_main2.avg:.4f}  test_penalty: {test_loss_penalty.avg:.4f}', end = "\r")

#                 self.test_main_list.append(test_loss_main.avg)
#                 self.test_main2_list.append(test_loss_main2.avg)
#                 self.test_penalty_list.append(test_loss_penalty.avg)
#                 self.log.info(f'[{epoch + 1:0{fep}}/{self.num_epoch}]  train_main: {train_loss_main.avg:.6e}  train_penalty: {train_loss_penalty.avg:.6e}  train_main2: {train_loss_main2.avg:.4f}  \
#                    train_penalty: {train_loss_penalty.avg:.6e}  test_main: {test_loss_main.avg:.6e}  test_main2: {test_loss_main2.avg:.6e}  test_penalty: {test_loss_penalty.avg:.6e}')

#                 # Additional test set
#                 data, condition = next(iter(self.test_generator))

#                 n = len(data)
#                 prior_z = self.generate_prior(n)
#                 x = data.to(self.device)
#                 # y = condition.to(self.device)
#                 fake_latent = self.encode_s(x)
#                 recon = self.decode(fake_latent)

#                 if len(self.tensorboard_dir) > 0:
#                     self.writer.add_scalar('test/main', test_loss_main.avg, epoch)
#                     self.writer.add_scalar('test/penalty', test_loss_penalty.avg, epoch)

#                     if self.lamb > 0:
#                         # Embedding
#                         for_embed1 = fake_latent.detach().to('cpu').numpy()
#                         for_embed2 = prior_z.detach().to('cpu').numpy()
#                         label = ['fake']*len(for_embed1) + ['prior']*len(for_embed2)
#                         self.writer.add_embedding(np.concatenate((for_embed1, for_embed2)), metadata = label, global_step = epoch)

#                         # Sample Generation
#                         test_dec = self.decode(prior_z).detach().to('cpu').numpy()
#                         self.writer.add_images('generated_sample', (test_dec[0:32])*0.5 + 0.5, epoch)

#                     # Reconstruction
#                     self.writer.add_images('reconstruction', (np.concatenate((x.detach().to('cpu').numpy()[0:16], recon.detach().to('cpu').numpy()[0:16])))*0.5 + 0.5, epoch)
#                     self.writer.flush()

#                 if len(self.save_img_path) > 0:
#                     save_sample_images('%s/recon' % self.save_img_path, epoch, (np.concatenate((x.detach().to('cpu').numpy()[0:32], recon.detach().to('cpu').numpy()[0:32])))*0.5 + 0.5)
#                     plt.close()
#                     if self.lamb > 0:
#                         # Sample Generation
#                         test_dec = self.decode(prior_z).detach().to('cpu').numpy()
#                         save_sample_images('%s/gen' % self.save_img_path, epoch, (test_dec[0:64])*0.5 + 0.5)
#                         plt.close()
                
#                 if self.save_best:
#                     obj = test_loss_main.avg + self.lamb * test_loss_penalty.avg
#                     if self.best_obj[1] > obj:
#                         self.best_obj[0] = epoch + 1
#                         self.best_obj[1] = obj
#                         self.save(self.save_path)
#                         self.log.info("model saved, obj: %.6e" % obj)
#                 else:
#                     self.save(self.save_path)
#                     # self.log.info("model saved at: %s" % self.save_path)
                
#             scheduler.step()

#             if len(self.save_state) > 0:
#                 save_dict = {
#                     'epoch':epoch + 1,
#                     'model_state_dict':self.state_dict(),
#                     'optimizer_state_dict':optimizer.state_dict()
#                 } 
#                 if len(self.lr_schedule) > 0:
#                     save_dict['scheduler'] = scheduler.state_dict()
#                 torch.save(save_dict, self.save_state)
            
#         if not self.validate_batch:
#             self.save(self.save_path)

#         self.log.info('Training Finished!')
#         self.log.info("Elapsed time: %.3fs" % (time.time() - start_time))

#         if len(self.tensorboard_dir) > 0:
#             self.writer.close()

# class SSWAE_GAN_dev(CWAE_GAN_abstract):
#     def __init__(self, cfg, log, device = 'cpu', verbose = 1):
#         super(SSWAE_GAN_dev, self).__init__(cfg, log, device, verbose)
#         self.yz_dim = int(cfg['train_info']['yz_dim'])
#         self.lamb2 = float(cfg['train_info']['lambda2'])
#         self.disc_loss = nn.BCEWithLogitsLoss()
        
#         data_class = getattr(dataset, cfg['train_info']['train_data'])
#         labeled = cfg['train_info'].getboolean('train_data_label')

#         labeled_class = cfg['train_info']['labeled_class'].replace(' ', '').split(',')
#         labeled_class = [int(i) for i in labeled_class]
#         unlabeled_class = cfg['train_info']['unlabeled_class'].replace(' ', '').split(',')
#         unlabeled_class = [int(i) for i in unlabeled_class]
#         test_class = cfg['train_info']['test_class'].replace(' ', '').split(',')
#         test_class = [int(i) for i in test_class]

#         self.batch_size1 = int(cfg['train_info']['batch_size1'])
#         self.batch_size2 = int(cfg['train_info']['batch_size2'])
#         batch_size = max(self.batch_size1, self.batch_size2)

#         self.train_data1 = data_class(cfg['path_info']['data_home2'], train = True, label = labeled, aux = [labeled_class, []])
#         self.train_generator1 = torch.utils.data.DataLoader(self.train_data1, self.batch_size1, num_workers = 5, shuffle = True, pin_memory=True, drop_last=True)

#         self.train_data2 = data_class(cfg['path_info']['data_home2'], train = True, label = labeled, aux = [labeled_class, unlabeled_class])
#         self.train_generator2 = torch.utils.data.DataLoader(self.train_data2, self.batch_size2, num_workers = 5, shuffle = True, pin_memory=True, drop_last=True)

#         self.test_data = data_class(cfg['path_info']['data_home2'], train = False, label = labeled, aux = [labeled_class, unlabeled_class + test_class])
#         self.test_generator = torch.utils.data.DataLoader(self.test_data, batch_size, num_workers = 5, shuffle = True, pin_memory=True, drop_last=False)

#         self.enc2 = nn.Identity()
#         self.enc_c = nn.Identity()
#         self.dec_c = nn.Identity()
#         try:
#             w = float(cfg['train_info']['classification_weight'])
#             self.bce = nn.BCEWithLogitsLoss(weight = torch.cat((torch.ones(self.y_dim - 1)/(self.y_dim - 1)*(1-w),torch.Tensor([w]))).to(self.device))
#         except:
#             self.bce = nn.BCEWithLogitsLoss()

#     def generate_prior(self, n):
#         return self.z_sampler(n, self.yz_dim + self.z_dim, device = self.device)

#     def encode_s(self, x):
#         xx = self.embed_data(x)
#         return torch.cat((self.enc2(torch.cat((self.enc_c(xx), xx), dim = 1)), self.enc(xx)), dim = 1)

#     def encode(self, x, y):
#         xx = self.embed_data(x)
#         return torch.cat((self.enc2(torch.cat((y, xx), dim = 1)), self.enc(xx)), dim = 1)
    
#     def decode(self, z):
#         return self.dec(z)

#     def decode_c(self, z):
#         return self.dec_c(z[:,0:self.yz_dim])
        
#     def forward(self, x, y):
#         return self.decode(self.encode(x, y))

#     def classify_loss(self, recon, y):
#         return self.bce(recon, y)

#     def train(self, resume = False):
#         self.train_main_list = []
#         self.train_main2_list = []
#         self.train_penalty_list = []
#         self.test_main_list = []
#         self.test_main2_list = []
#         self.test_penalty_list = []

#         for net in self.encoder_trainable:
#             net.train()
#         for net in self.decoder_trainable:
#             net.train()
            
#         if self.encoder_pretrain:
#             self.pretrain_encoder()
#             self.log.info('Pretraining Ended!')
            
#         if len(self.tensorboard_dir) > 0:
#             self.writer = SummaryWriter(self.tensorboard_dir)
            
#         optimizer_main = optim.Adam(sum([list(net.parameters()) for net in self.encoder_trainable], []) + sum([list(net.parameters()) for net in self.decoder_trainable], []), lr = self.lr, betas = (self.beta1, 0.999))
#         optimizer_adv = optim.Adam(sum([list(net.parameters()) for net in self.discriminator_trainable], []), lr = self.lr_adv, betas = (self.beta1_adv, 0.999))

#         start_epoch = 0
#         scheduler_main = self.lr_scheduler(optimizer_main)
#         scheduler_adv = self.lr_scheduler(optimizer_adv)

#         if resume:
#             checkpoint = torch.load(self.save_state)
#             start_epoch = checkpoint['epoch']
#             self.load_state_dict(checkpoint['model_state_dict'])
#             optimizer_main.load_state_dict(checkpoint['optimizer_main_state_dict'])
#             optimizer_adv.load_state_dict(checkpoint['optimizer_adv_state_dict'])
#             if len(self.lr_schedule) > 0:
#                 scheduler_main.load_state_dict(checkpoint['scheduler_main'])
#                 scheduler_adv.load_state_dict(checkpoint['scheduler_adv'])

#         self.log.info('------------------------------------------------------------')
#         self.log.info('Training Start!')
#         start_time = time.time()

#         iter_per_epoch = min(len(self.train_generator1), len(self.train_generator2))
#         n1 = self.batch_size1
#         n2 = self.batch_size2
        
#         for epoch in range(start_epoch, self.num_epoch):
#             # train_step
#             train_loss_main = inc_avg()
#             train_loss_main2 = inc_avg()
#             train_loss_penalty = inc_avg()

#             tr1 = iter(self.train_generator1)
#             tr2 = iter(self.train_generator2)

#             for i in range(iter_per_epoch):
#                 # with label
#                 data1, condition1 = next(tr1)
#                 data2, condition2 = next(tr2)
#                 prior_z = self.generate_prior(n1+n2)

#                 if self.lamb > 0:
#                     for net in self.discriminator_trainable:
#                         net.zero_grad()

#                     x1 = data1.to(self.device)
#                     y1 = condition1.to(self.device)
#                     fake_latent1 = self.encode(x1, y1)

#                     x2 = data2.to(self.device)
#                     fake_latent2 = self.encode_s(x2)

#                     fake_latent = torch.cat((fake_latent1, fake_latent2), dim = 0)

#                     adv = self.adv_loss(prior_z, fake_latent)
#                     obj_adv = self.lamb * adv
#                     obj_adv.backward()
#                     optimizer_adv.step()

#                 for net in self.encoder_trainable:
#                     net.zero_grad()
#                 for net in self.decoder_trainable:
#                     net.zero_grad()

#                 # with label

#                 x1 = data1.to(self.device)
#                 y1 = condition1.to(self.device)
#                 fake_latent1 = self.encode(x1, y1)
#                 recon = self.decode(fake_latent1)
#                 recon_y = self.decode_c(fake_latent1)
                
#                 loss1 = self.main_loss(x1, recon)
#                 loss2 = self.classify_loss(recon_y, y1)
#                 loss = loss1 + self.lamb2 * loss2

#                 # without label

#                 x2 = data2.to(self.device)
#                 y2 = condition2.to(self.device)
#                 fake_latent2 = self.encode_s(x2)
#                 recon2 = self.decode(fake_latent2)
#                 recon_y2 = self.decode_c(fake_latent2)

#                 los1 = self.main_loss(x2, recon2)
#                 los2 = self.classify_loss(recon_y2, y2)
#                 los = los1 + self.lamb2 * los2

#                 fake_latent = torch.cat((fake_latent1, fake_latent2), dim = 0)

#                 if self.lamb > 0:
#                     penalty = self.penalty_loss(fake_latent)
#                     obj = loss *(n1/(n1+n2))  + los *(n2/(n1+n2)) + self.lamb * penalty
#                 else:
#                     obj = loss *(n1/(n1+n2))  + los *(n2/(n1+n2))

#                 obj.backward()
#                 optimizer_main.step()
                
#                 train_loss_main.append((loss1 *(n1/(n1+n2)) + los1 *(n2/(n1+n2))).item(), n1+n2)
#                 train_loss_main2.append((loss2 *(n1/(n1+n2)) + los2 *(n2/(n1+n2))).item(), n1+n2)
#                 if self.lamb > 0:
#                     train_loss_penalty.append(penalty.item(), n1+n2)
                
#                 print(f'[{i+1:0{ftrep}}/{trep}]  train_main: {train_loss_main.avg:.4f}  train_main2: {train_loss_main2.avg:.4f}  train_penalty: {train_loss_penalty.avg:.4f}', end = "\r")

#             self.train_main_list.append(train_loss_main.avg)
#             self.train_main2_list.append(train_loss_main2.avg)
#             self.train_penalty_list.append(train_loss_penalty.avg)

#             if len(self.tensorboard_dir) > 0:
#                 self.writer.add_scalar('train/main', train_loss_main.avg, epoch)
#                 self.writer.add_scalar('train/penalty', train_loss_penalty.avg, epoch)
#                 if self.cfg['train_info'].getboolean('histogram'):
#                     for param_tensor in self.state_dict():
#                         self.writer.add_histogram(param_tensor, self.state_dict()[param_tensor].detach().to('cpu').numpy().flatten(), epoch)
            
#             # validation_step
#             test_loss_main = inc_avg()
#             test_loss_main2 = inc_avg()
#             test_loss_penalty = inc_avg()

#             if self.validate_batch:
#                 for i, (data, condition) in enumerate(self.test_generator):
#                     n = len(data)
#                     # test without label
#                     x = data.to(self.device)
#                     y = condition.to(self.device)
#                     fake_latent = self.encode_s(x)
#                     recon = self.decode(fake_latent)
#                     recon_y = self.decode_c(fake_latent)

#                     loss1 = self.main_loss(x, recon)
#                     loss2 = self.classify_loss(recon_y, y)

#                     test_loss_main.append(loss1.item(), n)
#                     test_loss_main2.append(loss2.item(), n)
#                     if self.lamb > 0:
#                         test_loss_penalty.append(self.penalty_loss(fake_latent).item(), n)
#                     print(f'[{i+1:0{fttep}}/{ttep}]  test_main: {test_loss_main.avg:.4f}  test_main2: {test_loss_main2.avg:.4f}  test_penalty: {test_loss_penalty.avg:.4f}', end = "\r")

#                 self.test_main_list.append(test_loss_main.avg)
#                 self.test_main2_list.append(test_loss_main2.avg)
#                 self.test_penalty_list.append(test_loss_penalty.avg)
#                 self.log.info(f'[{epoch + 1:0{fep}}/{self.num_epoch}]  train_main: {train_loss_main.avg:.6e}  train_penalty: {train_loss_penalty.avg:.6e}  train_main2: {train_loss_main2.avg:.4f}  \
#                    train_penalty: {train_loss_penalty.avg:.6e}  test_main: {test_loss_main.avg:.6e}  test_main2: {test_loss_main2.avg:.6e}  test_penalty: {test_loss_penalty.avg:.6e}')

#                 # Additional test set
#                 data, condition = next(iter(self.test_generator))

#                 n = len(data)
#                 prior_z = self.generate_prior(n)
#                 x = data.to(self.device)
#                 # y = condition.to(self.device)
#                 fake_latent = self.encode_s(x)
#                 recon = self.decode(fake_latent)

#                 if len(self.tensorboard_dir) > 0:
#                     self.writer.add_scalar('test/main', test_loss_main.avg, epoch)
#                     self.writer.add_scalar('test/penalty', test_loss_penalty.avg, epoch)

#                     if self.lamb > 0:
#                         # Embedding
#                         for_embed1 = fake_latent.detach().to('cpu').numpy()
#                         for_embed2 = prior_z.detach().to('cpu').numpy()
#                         label = ['fake']*len(for_embed1) + ['prior']*len(for_embed2)
#                         self.writer.add_embedding(np.concatenate((for_embed1, for_embed2)), metadata = label, global_step = epoch)

#                         # Sample Generation
#                         test_dec = self.decode(prior_z).detach().to('cpu').numpy()
#                         self.writer.add_images('generated_sample', (test_dec[0:32])*0.5 + 0.5, epoch)

#                     # Reconstruction
#                     self.writer.add_images('reconstruction', (np.concatenate((x.detach().to('cpu').numpy()[0:16], recon.detach().to('cpu').numpy()[0:16])))*0.5 + 0.5, epoch)
#                     self.writer.flush()

#                 if len(self.save_img_path) > 0:
#                     save_sample_images('%s/recon' % self.save_img_path, epoch, (np.concatenate((x.detach().to('cpu').numpy()[0:32], recon.detach().to('cpu').numpy()[0:32])))*0.5 + 0.5)
#                     plt.close()
#                     if self.lamb > 0:
#                         # Sample Generation
#                         test_dec = self.decode(prior_z).detach().to('cpu').numpy()
#                         save_sample_images('%s/gen' % self.save_img_path, epoch, (test_dec[0:64])*0.5 + 0.5)
#                         plt.close()
                
#                 if self.save_best:
#                     obj = test_loss_main.avg + self.lamb * test_loss_penalty.avg
#                     if self.best_obj[1] > obj:
#                         self.best_obj[0] = epoch + 1
#                         self.best_obj[1] = obj
#                         self.save(self.save_path)
#                         self.log.info("model saved, obj: %.6e" % obj)
#                 else:
#                     self.save(self.save_path)
#                     # self.log.info("model saved at: %s" % self.save_path)
                
#             scheduler_main.step()
#             if self.lamb > 0:
#                 scheduler_adv.step()

#             if len(self.save_state) > 0:
#                 save_dict = {
#                     'epoch':epoch + 1,
#                     'model_state_dict':self.state_dict(),
#                     'optimizer_main_state_dict':optimizer_main.state_dict(),
#                     'optimizer_adv_state_dict':optimizer_adv.state_dict()
#                 } 
#                 if len(self.lr_schedule) > 0:
#                     save_dict['scheduler_main'] = scheduler_main.state_dict()
#                     save_dict['scheduler_adv'] = scheduler_adv.state_dict()
#                 torch.save(save_dict, self.save_state)
            
#         if not self.validate_batch:
#             self.save(self.save_path)

#         self.log.info('Training Finished!')
#         self.log.info("Elapsed time: %.3fs" % (time.time() - start_time))

#         if len(self.tensorboard_dir) > 0:
#             self.writer.close()