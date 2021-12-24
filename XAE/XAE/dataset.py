import torch
import PIL
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import pickle

class ToyDataset(torch.utils.data.Dataset): 
    def __init__(self, data_mat):
        self.data = data_mat
        self.z_dim = len(data_mat[0])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx): 
        return self.data[idx]

class CelebA(torch.utils.data.Dataset):
    def __init__(self, data_home, train = True, label = False):
        self.data_home = '%s/img_align_celeba' % data_home
        if train:
            self.x_list = np.load('%s/x_list_train.npy' % data_home)
        else:
            self.x_list = np.load('%s/x_list_test.npy' % data_home)
        # self.list_attr = pd.read_csv('%s/list_attr_celeba.csv' % data_home)
        # self.list_bbox = pd.read_csv('%s/list_bbox_celeba.csv' % data_home)
        # self.list_eval = pd.read_csv('%s/list_eval_partition.csv' % data_home)

        # closecrop
        self.transform = transforms.Compose([
            transforms.CenterCrop((140, 140)),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),])
        
    def __len__(self):
        return len(self.x_list)
    
    def __getitem__(self, idx):
        with Image.open('%s/%s' % (self.data_home, self.x_list[idx])) as im:
            return 2.0*self.transform(im) - 1.0
        
class MNIST(torch.utils.data.Dataset):
    def __init__(self, data_home, train = True, label = False, output_channels = 1, portion = 1.0):
        self.label = label
        self.output_channels = output_channels
        if train:
            self.data = np.loadtxt('%s/mnist_train.csv' % data_home, delimiter=',', skiprows = 1)
        else:
            self.data = np.loadtxt('%s/mnist_test.csv' % data_home, delimiter=',', skiprows = 1)

        if portion < 1.0:
            k = int(np.shape(self.data)[0]*portion)
            self.data[k:, 0] = 10

        self.code = torch.from_numpy(np.concatenate([np.eye(10), np.zeros((1,10))], axis = 0)).type(torch.float32)
        
    def __len__(self):
        return np.shape(self.data)[0]
    
    def __getitem__(self, idx):
        if self.label:
            if self.output_channels > 1:
                return [torch.from_numpy(2. * (self.data[idx, 1:785]/255) - 1.).reshape((1,28,28)).type(torch.float32).repeat((self.output_channels,1,1)), self.code[self.data[idx, 0].astype(np.int)]]
            else:
                return [torch.from_numpy(2. * (self.data[idx, 1:785]/255) - 1.).reshape((1,28,28)).type(torch.float32), self.code[self.data[idx, 0].astype(np.int)]]
        else:
            if self.output_channels > 1:
                return torch.from_numpy(2. * (self.data[idx, 1:785]/255) - 1.) .reshape((1,28,28)).type(torch.float32).repeat((self.output_channels,1,1))
            else:
                return torch.from_numpy(2. * (self.data[idx, 1:785]/255) - 1.) .reshape((1,28,28)).type(torch.float32)

class rmMNIST(MNIST):
    def __init__(self, data_home, train = True, label = False, output_channels = 1, aux = None, portion = 1.0):
        self.label = label
        self.output_channels = output_channels
        self.portion = portion
        
        if train:
            self.data = np.loadtxt('%s/mnist_train.csv' % data_home, delimiter=',', skiprows = 1)
        else:
            self.data = np.loadtxt('%s/mnist_test.csv' % data_home, delimiter=',', skiprows = 1)
        
        if aux is not None:
            lab_ind = [[] for i in range(10)]
            # classify all data by digit
            for i in range(len(self.data)):
                lab_ind[int(self.data[i,0])].append(i)
            # count all counts for each digit
            all_ind = []
            recode = 0
            unknown = 0
            for i in aux[0]:
                k = int(len(lab_ind[i])*self.portion)
                using_digit_ind = lab_ind[i][0:k]
                self.data[using_digit_ind, 0] = recode
                all_ind += using_digit_ind
                recode += 1
            for i in aux[1]:
                k = int(len(lab_ind[i])*self.portion)
                using_digit_ind = lab_ind[i][0:k]
                self.data[using_digit_ind, 0] = recode
                all_ind += using_digit_ind
                unknown = 1
            self.data = self.data[all_ind]
            self.code = torch.from_numpy(np.eye(recode + unknown)).type(torch.float32)
        else:
            self.code = torch.from_numpy(np.eye(10)).type(torch.float32)

class eYaleB(torch.utils.data.Dataset):
    def __init__(self, data_home, train = True, label = True, output_channels = 1):
        self.data = None
        self.output_channels = output_channels
        self.label = label
        if train:
            with open('%s/YaleBFaceTrain.dat' % data_home, 'rb') as f:
                self.data = pickle.load(f)
        else:
            with open('%s/YaleBFaceTest.dat' % data_home, 'rb') as f:
                self.data = pickle.load(f)

        self.code = torch.from_numpy(np.eye(28)).type(torch.float32)
        self.code2 = torch.from_numpy(np.eye(10)).type(torch.float32)
        
    def __len__(self):
        return (self.data['image'].shape)[0]
    
    def __getitem__(self, idx):
        if self.label:
            if self.output_channels > 1:
                return [torch.from_numpy(2. * (self.data['image'][idx, :]/255) - 1.).reshape((1,128,128)).type(torch.float32).repeat((self.output_channels,1,1)), torch.cat((self.code[self.data['person'][idx].astype(np.int)], self.code2[self.data['pose'][idx].astype(np.int)], torch.Tensor([self.data['azimuth'][idx]/180.0]), torch.Tensor([self.data['elevation'][idx]/90.0]))).type(torch.float32)]
            else: 
                return [torch.from_numpy((2. * (self.data['image'][idx, :]/255) - 1.).reshape((1,128,128))).type(torch.float32), 
                    torch.cat((self.code[self.data['person'][idx].astype(np.int)], self.code2[self.data['pose'][idx].astype(np.int)], torch.Tensor([self.data['azimuth'][idx]/180.0]), torch.Tensor([self.data['elevation'][idx]/90.0])))]
        else:
            if self.output_channels > 1:
                return torch.from_numpy(2. * (self.data['image'][idx, :]/255) - 1.).reshape((1,128,128)).type(torch.float32).repeat((self.output_channels,1,1))
            else: 
                return torch.from_numpy((2. * (self.data['image'][idx, :]/255) - 1.).reshape((1,128,128))).type(torch.float32)
