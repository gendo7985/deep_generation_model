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

class WAE_MMD_abstract(XAE_abstract):
    def __init__(self, cfg, log, device = 'cpu', verbose = 1):
        super(WAE_MMD_abstract, self).__init__(cfg, log, device, verbose)
        self.main_loss = nn.MSELoss()

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

    def train_loss(self, x, y):
        return self.main_loss(x,y)
        
    def penalty_loss(self, x, y, n):
        return (self.k(x,x, False) + self.k(y,y, False))/(n*(n-1)) - 2*self.k(x,y, True)/(n*n)

class CWAE_MMD_abstract(CXAE_abstract):
    def __init__(self, cfg, log, device = 'cpu', verbose = 1):
        super(CWAE_MMD_abstract, self).__init__(cfg, log, device, verbose)
        self.main_loss = nn.MSELoss()

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

    def train_loss(self, x, y):
        return self.main_loss(x,y)
        
    def penalty_loss(self, x, y, n):
        return (self.k(x,x, False) + self.k(y,y, False))/(n*(n-1)) - 2*self.k(x,y, True)/(n*n)

class WAE_GAN_abstract(XAE_adv_abstract):
    def __init__(self, cfg, log, device = 'cpu', verbose = 1):
        super(WAE_GAN_abstract, self).__init__(cfg, log, device, verbose)
        self.main_loss = nn.MSELoss()
        self.disc_loss = nn.BCEWithLogitsLoss()

    def train_loss(self, x, y):
        return self.main_loss(x,y)

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
        self.main_loss = nn.MSELoss()
        self.disc_loss = nn.BCEWithLogitsLoss()

    def train_loss(self, x, y):
        return self.main_loss(x,y)

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
        self.recon_loss = nn.BCEWithLogitsLoss(reduction='none')
        
    def train(self, resume = False):
        self.train_loss_list = []
        self.test_loss_list = []
            
        if self.encoder_pretrain:
            self.pretrain_encoder()
            self.log.info('Pretraining Ended!')
            
        if len(self.tensorboard_dir) > 0:
            self.writer = SummaryWriter(self.tensorboard_dir)
            
        optimizer = optim.Adam(sum([list(net.parameters()) for net in self.encoder_trainable], []) + sum([list(net.parameters()) for net in self.decoder_trainable], []), lr = self.lr, betas = (self.beta1, 0.999))

        start_epoch = 0
        scheduler = self.lr_scheduler(optimizer)
        if self.lr_schedule is "manual":
            lamb = lambda e: 1.0 * (0.5 ** (e >= 30)) * (0.2 ** (e >= 50)) * (0.1 ** (e >= 100))
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lamb)

        if resume:
            checkpoint = torch.load(self.save_state)
            start_epoch = checkpoint['epoch']
            self.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if self.lr_schedule is "manual":
                scheduler.load_state_dict(checkpoint['scheduler'])

        self.log.info('------------------------------------------------------------')
        self.log.info('Training Start!')
        start_time = time.time()
        
        for epoch in range(start_epoch, self.num_epoch):
            train_loss = inc_avg()
            
            for net in self.encoder_trainable:
                net.train()
            for net in self.decoder_trainable:
                net.train()
            
            for i, data in enumerate(self.train_generator, 0):
                for net in self.encoder_trainable:
                    net.zero_grad()
                for net in self.decoder_trainable:
                    net.zero_grad()

                x = data.to(self.device)*0.5+0.5

                z = self.enc(x)
                mu = self.mu(z)
                logvar = self.logvar(z)
                
                recon = self.decode(self.reparameterize(mu, logvar))
                
                logpx_z = torch.sum(self.recon_loss(recon,x), axis=[1,2,3])
                D_KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), axis=1)

                total_loss = torch.mean(logpx_z + D_KL)
                total_loss.backward()
                optimizer.step()
                
                train_loss.append(total_loss.item(), len(data))
                print('[%i/%i]\ttrain_mse: %.4f\r' % (i+1, len(self.train_generator), train_loss_mse.avg))

            self.train_loss_list.append(train_loss.avg)
    
            if len(self.tensorboard_dir) > 0:
                self.writer.add_scalar('train/BCE', train_loss.avg, epoch)
                if self.cfg['train_info'].getboolean('histogram'):
                    for param_tensor in self.state_dict():
                        self.writer.add_histogram(param_tensor, self.state_dict()[param_tensor].detach().to('cpu').numpy().flatten(), epoch)

            # validation_step
            test_loss = inc_avg()
            
            for net in self.encoder_trainable:
                    net.eval()
            for net in self.decoder_trainable:
                    net.eval()

            if self.validate_batch:
                for i, data in enumerate(self.test_generator):
                    with torch.no_grad():
                        x = data.to(self.device)*.5+.5
                        z = self.enc(x).detach()
                        
                        mu = self.mu(z)
                        logvar = self.logvar(z)
                        
                        recon = self.decode(self.reparameterize(mu, logvar)).detach()
                        logpx_z = torch.sum(self.recon_loss(recon,x), axis=[1,2,3])
                        D_KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), axis=1)

                        total_loss = torch.mean(logpx_z + D_KL)
                        test_loss.append(total_loss.item(), len(data))

                self.test_loss_list.append(test_loss.avg)
                self.log.info('[%d/%d]\ttrain_loss: %.6e\ttest_loss: %.6e\t'
                      % (epoch + 1, self.num_epoch, train_loss.avg, test_loss.avg))
                
                # Additional test set
                data = next(iter(self.test_generator))
                
                prior_z = self.generate_prior(len(data))
                x = data.to(self.device)*.5+.5
                z = self.enc(x).detach()
                mu = self.mu(z)
                logvar = self.logvar(z)
                recon = self.decode(self.reparameterize(mu, logvar)).detach()
                
                if len(self.tensorboard_dir) > 0:
                    self.writer.add_scalar('test/BCE', test_loss.avg, epoch)

                    # Reconstruction
                    self.writer.add_images('reconstruction', (np.concatenate((x.to('cpu').numpy()[0:16], sigmoid(recon.to('cpu').numpy())[0:16]))), epoch)
                    
                    # Sample Generation
                    test_dec = self.dec(prior_z).detach().to('cpu').numpy()
                    self.writer.add_images('generation', sigmoid(test_dec)[0:32])
                    
                    self.writer.flush()
                
                if len(self.save_img_path) > 0:
                    # Reconstruction
                    save_sample_images('%s/recon' % self.save_img_path, epoch, np.concatenate((x.to('cpu').numpy()[0:32], sigmoid(recon.to('cpu').numpy())[0:32])))
                    plt.close()
                    
                    # Sample Generation
                    test_dec = self.dec(prior_z).detach().to('cpu').numpy()
                    save_sample_images('%s/gen' % self.save_img_path, epoch, sigmoid(test_dec)[0:64])
                    plt.close()
                        
                if self.save_best:
                    obj = test_loss.avg
                    if self.best_obj[1] > obj:
                        self.best_obj[0] = epoch + 1
                        self.best_obj[1] = obj
                        self.save(self.save_path)
                        self.log.info("model saved, obj: %.6e" % obj)
                else:
                    self.save(self.save_path)
                    # self.log.info("model saved at: %s" % self.save_path)
                
            if self.lr_schedule is not None:
                scheduler.step()

            if self.save_state is not None:
                save_dict = {
                    'epoch':epoch + 1,
                    'model_state_dict':self.state_dict(),
                    'optimizer_state_dict':optimizer.state_dict()
                } 
                if self.lr_schedule is not None:
                    save_dict['scheduler'] = scheduler.state_dict()
                torch.save(save_dict, self.save_state)
            
        if not self.validate_batch:
            self.save(self.save_path)

        self.log.info('Training Finished!')
        self.log.info("Elapsed time: %.3fs" % (time.time() - start_time))

        if len(self.tensorboard_dir) > 0:
            self.writer.close()
        
class CVAE_abstract(VAE_abstract):
    def __init__(self, cfg, log, device = 'cpu', verbose = 1):
        super(CVAE_abstract, self).__init__(cfg, log, device, verbose)
        
        data_class = getattr(dataset, cfg['train_info']['train_data'])
        self.train_data =  data_class(cfg['path_info']['data_home'], train = True, label = True)
        self.test_data = data_class(cfg['path_info']['data_home'], train = False, label = True)
        if cfg['train_info'].getboolean('replace'):
            it = int(cfg['train_info']['iter_per_epoch'])
            train_sampler = torch.utils.data.RandomSampler(self.train_data, replacement = True, num_samples = self.batch_size * it)
            self.train_generator = torch.utils.data.DataLoader(self.train_data, self.batch_size, num_workers = 5, sampler = train_sampler, pin_memory=True)
        else:
            self.train_generator = torch.utils.data.DataLoader(self.train_data, self.batch_size, num_workers = 5, shuffle = True, pin_memory=True, drop_last=True)

        self.test_generator = torch.utils.data.DataLoader(self.test_data, self.batch_size, num_workers = 5, shuffle = True, pin_memory=True, drop_last=True)
        
        self.y_sampler = getattr(sampler, cfg['train_info']['y_sampler']) # generate prior
        self.y_dim = int(cfg['train_info']['y_dim'])

        # Abstract part
        self.embed_data = nn.Identity()
        self.embed_condition = nn.Identity()
        
    def encode(self, x, y):
        if self.prob_enc:
            z = self.enc(torch.cat((self.embed_data(x), self.embed_condition(y)), dim = 1))
            return self.reparameterize(self.mu(z), self.logvar(z))
        else:
            return self.enc(torch.cat((self.embed_data(x), self.embed_condition(y)), dim = 1))
            
    def forward(self, x, y):
        return self.decode(torch.cat((self.encode(x,y), y), dim = 1))
        
    def train(self, resume = False):
        self.train_loss_list = []
        self.test_loss_list = []
        
        if self.encoder_pretrain:
            self.pretrain_encoder()
            self.log.info('Pretraining Ended!')
            
        if len(self.tensorboard_dir) > 0:
            self.writer = SummaryWriter(self.tensorboard_dir)
            
        optimizer = optim.Adam(sum([list(net.parameters()) for net in self.encoder_trainable], []) + sum([list(net.parameters()) for net in self.decoder_trainable], []), lr = self.lr, betas = (self.beta1, 0.999))
        
        start_epoch = 0
        scheduler = self.lr_scheduler(optimizer)
        if self.lr_schedule is "manual":
            lamb = lambda e: 1.0 * (0.5 ** (e >= 30)) * (0.2 ** (e >= 50)) * (0.1 ** (e >= 100))
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lamb)

        if resume:
            checkpoint = torch.load(self.save_state)
            start_epoch = checkpoint['epoch']
            self.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if len(self.lr_schedule) > 0:
                scheduler.load_state_dict(checkpoint['scheduler'])

        self.log.info('------------------------------------------------------------')
        self.log.info('Training Start!')
        start_time = time.time()
        
        for epoch in range(start_epoch, self.num_epoch):
            train_loss = inc_avg()
            
            for net in self.encoder_trainable:
                    net.train()
            for net in self.decoder_trainable:
                    net.train()
            
            for i, (data, condition) in enumerate(self.train_generator, 0):
                for net in self.encoder_trainable:
                    net.zero_grad()
                for net in self.decoder_trainable:
                    net.zero_grad()

                x = data.to(self.device)*.5+.5
                y = condition.to(self.device)
                
                z = self.enc(torch.cat((self.embed_data(x), self.embed_condition(y)), dim = 1))
                mu = self.mu(z)
                logvar = self.logvar(z)

                recon = self.decode(torch.cat((self.reparameterize(mu, logvar),y), dim = 1))
                logpx_z = torch.sum(self.recon_loss(recon,x), axis=[1,2,3])
                D_KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), axis=1)

                total_loss = torch.mean(logpx_z + D_KL)
                total_loss.backward()
                optimizer.step()
                
                train_loss.append(total_loss.item(), len(data[0]))

            self.train_loss_list.append(train_loss.avg)
            
            if len(self.tensorboard_dir) > 0:
                self.writer.add_scalar('train/BCE', train_loss.avg, epoch)
                if self.cfg['train_info'].getboolean('histogram'):
                    for param_tensor in self.state_dict():
                        self.writer.add_histogram(param_tensor, self.state_dict()[param_tensor].detach().to('cpu').numpy().flatten(), epoch)

            # validation_step
            test_loss = inc_avg()
            
            for net in self.encoder_trainable:
                    net.eval()
            for net in self.decoder_trainable:
                    net.eval()

            if self.validate_batch:
                for i, (data, condition) in enumerate(self.test_generator):
                    with torch.no_grad():
                        x = data.to(self.device)*.5+.5
                        y = condition.to(self.device)

                        z = self.enc(torch.cat((self.embed_data(x), self.embed_condition(y)), dim = 1)).detach()
                        
                        mu = self.mu(z).detach()
                        logvar = self.logvar(z).detach()

                        recon = self.decode(torch.cat((self.reparameterize(mu, logvar),y), dim = 1)).detach()
                        logpx_z = torch.sum(self.recon_loss(recon,x), axis=[1,2,3])
                        D_KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), axis=1)

                        total_loss = torch.mean(logpx_z + D_KL)
                        test_loss.append(total_loss.item(), len(data[0]))

                self.test_loss_list.append(test_loss.avg)
                self.log.info('[%d/%d]\ttrain_loss: %.6e\ttest_loss: %.6e\t'
                      % (epoch + 1, self.num_epoch, train_loss.avg, test_loss.avg))
                
                # Additional test set
                data, condition = next(iter(self.test_generator))

                x = data.to(self.device)*.5+.5
                y = condition.to(self.device)
                prior_z = self.generate_prior(len(data))
                fake_latent = torch.cat((self.encode(x, y), y), dim = 1).detach()
                recon = self.decode(fake_latent).detach()
                
                if len(self.tensorboard_dir) > 0:
                    self.writer.add_scalar('test/BCE', test_loss.avg, epoch)
                    
                    # Reconstruction
                    self.writer.add_images('reconstruction', (np.concatenate((x.to('cpu').numpy()[0:16], sigmoid(recon.to('cpu').numpy())[0:16]))), epoch)
                    
                    # Sample Generation
                    test_dec = self.dec(torch.cat((prior_z,y), dim=1)).detach().to('cpu').numpy()
                    self.writer.add_images('generation', sigmoid(test_dec)[0:32])
                    
                    self.writer.flush()
                               
                if len(self.save_img_path) > 0:
                    # Reconstruction
                    save_sample_images('%s/recon' % self.save_img_path, epoch, np.concatenate((x.to('cpu').numpy()[0:32], sigmoid(recon.to('cpu').numpy())[0:32])))
                    plt.close()
                    
                    # Sample Generation
                    test_dec = self.dec(torch.cat((prior_z,y), dim=1)).detach().to('cpu').numpy()
                    save_sample_images('%s/gen' % self.save_img_path, epoch, sigmoid(test_dec)[0:64])
                    plt.close()
                
                if self.save_best:
                    obj = test_loss.avg
                    if self.best_obj[1] > obj:
                        self.best_obj[0] = epoch + 1
                        self.best_obj[1] = obj
                        self.save(self.save_path)
                        self.log.info("model saved, obj: %.6e" % obj)
                else:
                    self.save(self.save_path)
                
            if self.lr_schedule is not None:
                scheduler.step()

            if self.save_state is not None:
                save_dict = {
                    'epoch':epoch + 1,
                    'model_state_dict':self.state_dict(),
                    'optimizer_state_dict':optimizer.state_dict()
                } 
                if self.lr_schedule is not None:
                    save_dict['scheduler'] = scheduler.state_dict()
                torch.save(save_dict, self.save_state)
            
        if not self.validate_batch:
            self.save(self.save_path)

        self.log.info('Training Finished!')
        self.log.info("Elapsed time: %.3fs" % (time.time() - start_time))

        if len(self.tensorboard_dir) > 0:
            self.writer.close()

class SCWAE_GAN_abstract(WAE_GAN_abstract):
    def __init__(self, cfg, log, device = 'cpu', verbose = 1):
        super(SCWAE_GAN_abstract, self).__init__(cfg, log, device, verbose)
        self.y_sampler = getattr(sampler, cfg['train_info']['y_sampler']) # generate prior
        self.y_dim = int(cfg['train_info']['y_dim'])

        # Abstract part
        self.embed_data = nn.Identity()
        self.embed_condition = nn.Identity()

    def encode(self, x, y):
        if self.prob_enc:
            z = self.enc(torch.cat((self.embed_data(x), self.embed_condition(y)), dim = 1))
            return reparameterize(self.mu(z), self.logvar(z))
        else:
            return self.enc(torch.cat((self.embed_data(x), self.embed_condition(y)), dim = 1))
            
    def forward(self, x, y):
        return self.decode(torch.cat((self.encode(x,y), self.embed_condition(y)), dim = 1))

    def generate_prior(self, n):
        return torch.cat((self.z_sampler(n, self.z_dim, device = self.device), self.y_sampler(n, self.y_dim, device = self.device)), dim = 1)

    def pretrain_encoder(self):
        optimizer = optim.Adam(sum([list(net).parameters() for net in self.encoder_trainable], []), lr = self.lr, betas = (self.beta1, 0.999))
        mse = nn.MSELoss()

        for net in self.encoder_trainable:
            net.train()
        
        self.log.info('------------------------------------------------------------')
        self.log.info('Pretraining Start!')
        
        cur_step = 0
        break_ind = False
        while True:
            for i, (data, condition) in enumerate(self.pretrain_generator):
                for net in self.encoder_trainable:
                    net.zero_grad()

                cur_step = cur_step + 1
                pz = torch.cat((self.z_sampler(len(data), self.z_dim, device = self.device), self.y_sampler(len(data), self.y_dim, device = self.device)), dim = 1)
                x = data.to(self.device)
                y = condition.to(self.device)
                qz = self.encode(x, y)

                qz_mean = torch.mean(qz, dim = 0)
                pz_mean = torch.mean(pz, dim = 0)

                qz_cov = torch.mean(torch.matmul((qz - qz_mean).unsqueeze(2), (qz - qz_mean).unsqueeze(1)), dim = 0)
                pz_cov = torch.mean(torch.matmul((pz - pz_mean).unsqueeze(2), (pz - pz_mean).unsqueeze(1)), dim = 0)

                loss = mse(pz_mean, qz_mean) + mse(pz_cov, qz_cov)

                loss.backward()
                optimizer.step()
                
                # train_loss_mse.append(loss.item(), len(data))
                if loss.item() > 0.1 or cur_step < 10:
                    print('train_mse: %.4f at %i step' % (loss.item(), cur_step), end = "\r")
                else:
                    self.log.info('train_mse: %.4f at %i step' % (loss.item(), cur_step))
                    break_ind = True
                    break
                    
            if break_ind or cur_step >= self.encoder_pretrain_step:
                break

    def train(self, resume = False):
        self.train_mse_list = []
        self.train_penalty_list = []
        self.test_mse_list = []
        self.test_penalty_list = []

        for net in self.encoder_trainable:
            net.train()
        for net in self.decoder_trainable:
            net.train()
        for net in self.discriminator_trainable:
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

        self.log.info('------------------------------------------------------------')
        self.log.info('Training Start!')
        start_time = time.time()
        
        for epoch in range(start_epoch, self.num_epoch):
            # train_step
            train_loss_mse = inc_avg()
            train_loss_penalty = inc_avg()
            
            for i, (data, condition) in enumerate(self.train_generator):
                n = len(data)
                prior_z = self.generate_prior(n)
                x = data.to(self.device)
                y = condition.to(self.device)

                if self.lamb > 0:
                    for net in self.discriminator_trainable:
                        net.zero_grad()

                    fake_latent = torch.cat((self.encode(x, y), self.embed_label(y)), dim = 1)
                    adv = self.adv_loss(prior_z, fake_latent)
                    obj_adv = self.lamb * adv
                    obj_adv.backward()
                    optimizer_adv.step()

                for net in self.encoder_trainable:
                    net.zero_grad()
                for net in self.decoder_trainable:
                    net.zero_grad()

                fake_latent = torch.cat((self.encode(x, y), self.embed_label(y)), dim = 1)
                recon = self.decode(fake_latent)
                
                loss = self.recon_loss(x, recon)
                if self.lamb > 0:
                    penalty = self.penalty_loss(fake_latent)
                    obj_main = loss + self.lamb * penalty
                else:
                    obj_main = loss
                obj_main.backward()
                optimizer_main.step()
                
                train_loss_mse.append(loss.item(), n)
                if self.lamb > 0:
                    train_loss_penalty.append(penalty.item(), n)
                
                print('[%i/%i]\ttrain_mse: %.4f\ttrain_penalty: %.4f' % (i+1, len(self.train_generator), train_loss_mse.avg, train_loss_penalty.avg), 
                      end = "\r")

            self.train_mse_list.append(train_loss_mse.avg)
            self.train_penalty_list.append(train_loss_penalty.avg)

            if len(self.tensorboard_dir) > 0:
                self.writer.add_scalar('train/MSE', train_loss_mse.avg, epoch)
                self.writer.add_scalar('train/penalty', train_loss_penalty.avg, epoch)
                if self.cfg['train_info'].getboolean('histogram'):
                    for param_tensor in self.state_dict():
                        self.writer.add_histogram(param_tensor, self.state_dict()[param_tensor].detach().to('cpu').numpy().flatten(), epoch)
            
            # validation_step
            test_loss_mse = inc_avg()
            test_loss_penalty = inc_avg()

            if self.validate_batch:
                for i, (data, condition) in enumerate(self.test_generator):

                    n = len(data)

                    prior_z = self.generate_prior(n)
                    x = data.to(self.device)
                    y = condition.to(self.device)
                    fake_latent = torch.cat((self.encode(x, y), self.embed_label(y)), dim = 1)
                    penalty = self.penalty_loss(fake_latent)
                    # adv = self.adv_loss(prior_z, fake_latent)
                    recon = self.decode(fake_latent)

                    test_loss_mse.append(self.recon_loss(x, recon).item(), n)
                    if self.lamb > 0:
                        test_loss_penalty.append(penalty.item(), n)
                    print('[%i/%i]\ttest_mse: %.4f\ttest_penalty: %.4f' % (i, len(self.test_generator), test_loss_mse.avg, test_loss_penalty.avg), end = "\r")

                self.test_mse_list.append(test_loss_mse.avg)
                self.test_penalty_list.append(test_loss_penalty.avg)
                
                self.log.info('[%d/%d]\ttrain_mse: %.6e\ttrain_penalty: %.6e\ttest_mse: %.6e\ttest_penalty: %.6e'
                      % (epoch + 1, self.num_epoch, train_loss_mse.avg, train_loss_penalty.avg, test_loss_mse.avg, test_loss_penalty.avg))

                # Additional test set
                data, condition = next(iter(self.test_generator))

                x = data.to(self.device)
                y = condition.to(self.device)
                prior_z = self.generate_prior(len(data))
                fake_latent = torch.cat((self.encode(x, y), self.embed_label(y)), dim = 1).detach()
                recon = self.decode(fake_latent).detach()

                if len(self.tensorboard_dir) > 0:
                    self.writer.add_scalar('test/MSE', test_loss_mse.avg, epoch)
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
                    save_sample_images('%s/recon' % self.save_img_path, epoch, (np.concatenate((x.to('cpu').numpy()[0:32], recon.to('cpu').numpy()[0:32])))*0.5 + 0.5)
                    plt.close()
                    if self.lamb > 0:
                        # Sample Generation
                        test_dec = self.decode(prior_z).detach().to('cpu').numpy()
                        save_sample_images('%s/gen' % self.save_img_path, epoch, (test_dec[0:64])*0.5 + 0.5)
                        plt.close()
                
                if self.save_best:
                    obj = test_loss_mse.avg + self.lamb * test_loss_penalty.avg
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