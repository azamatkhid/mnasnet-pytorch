import os
import torch
import numpy as np

import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.data.sampler import SubsetRandomSampler
from torchsummary import summary

from model import MnasNet

import hydra
from hydra import utils

class Application:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset = getattr(datasets,cfg.dataset.upper())
        self.data_dir = os.path.join(utils.get_original_cwd(),"./data") 
        self.device = ("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")

        self.train_transforms = transforms.Compose([transforms.Resize((224, 224), interpolation = 2),
            transforms.Pad(4),
            transforms.RandomHorizontalFlip(p = 0.5),
            transforms.RandomCrop((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.5, 0.5, 0.5],
                std = [0.5, 0.5, 0.5])])

        self.test_transforms = transforms.Compose([transforms.Resize((224, 224), interpolation = 2),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.5, 0.5, 0.5],
                std = [0.5, 0.5, 0.5])])

        self.net = None
        
    def build(self):
        '''
        this is for inhouse model implementations
        self.net = MnasNet()

        if torch.cuda.device_count() > 1:
            self.net = nn.DataParallel(self.net)
            print(f"Number of GPUs : {torch.cuda.device_count()}")

        self.net.to(self.device)
        if self.cfg.verbose and torch.cuda.device_count() <= 1:
            summary(self.net, (3, 224, 224))
        '''
        pass

    def train(self):
        self.net.train()
        self._check_dirs()
        self._load_data("train")

        pass

    def test(self):

        pass

    def _validation(self):

        pass

    def _check_dirs(self):
        if not os.path.exists(self.cfg.log_dir):
            os.mkdir(self.cfg.log_dir)
        if not os.path.exists(self.cfg.ckpts_dir):
            os.mkdir(self.cfg.ckpts_dir)

    def _load_data(self, *args):
        if args[0] == "train":
            data = self.dataset(root = self.data_dir, 
                    train = True, download = True, transform = self.train_transforms)
            data_size = len(data)
            indices = list(range(data_size))
            valid_ratio = 0.1
            split = int(np.floor(valid_ratio * data_size))
            train_idx, valid_idx = indices[split:], indices[:split]
            train_sampler = SubsetRandomSampler(train_idx)
            valid_sampler = SubsetRandomSampler(valid_idx)

            self.train_data = torch.utils.data.DataLoader(data,
                    batch_size = self.cfg.batch_size,
                    sampler = train_sampler,
                    num_workers = 1)

            self.valid_data = torch.utils.data.DataLoader(data,
                    batch_size = self.cfg.batch_size,
                    sampler = train_sampler,
                    num_workers = 1)
        elif args[0] == "test":
            data = self.dataset(root = self.data_dir,
                    train = False, download = True, transforms = self.test_transforms)
            self.test_data = torch.utils.data.DataLoader(data,
                    batch_size = self.cfg.batch_size,
                    shuffle = False,
                    num_workers = 1)

@hydra.main(config_path="./default.yaml")
def main(cfg):
    app=Application(cfg.parameters)
    app.build()
    app.train()


if __name__=="__main__":
    main()
