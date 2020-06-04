import os
import torch
import numpy as np

import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from model import MnasNet

import hydra
from hydra import utils

class Application:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset = getattr(datasets,cfg.dataset.upper())
        self.data_dir = os.path.join(utils.get_original_cwd(),"data") 
        self.device = ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")

        self.train_transforms = transforms.Compose([transforms.Resize((224, 224), interpolation=2),
            transforms.Pad(4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.5, 0.5, 0.5],
                std = [0.5, 0.5, 0.5])])

        self.test_transforms = transforms.Compose([transforms.Resize((224, 224), interpolation=2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

        print(f"[*] Device: {self.device}")
        print(f"[*] Path: {self.data_dir}")

        self.net = None
        
    def build(self, criterion=nn.CrossEntropyLoss, optimizer=torch.optim.RMSprop):
        self.net = MnasNet(3, self.cfg, training=True)
        self.net = nn.DataParallel(self.net)
        
        if torch.cuda.device_count() > 1:
            print(f"[*] Number of GPUs : {torch.cuda.device_count()}")

        self.net.to(self.device)
        if self.cfg.verbose and torch.cuda.device_count() <= 1:
            summary(self.net, (3, 224, 224))

        self.criterion = criterion()
        self.optimizer = optimizer(self.net.parameters(), lr=self.cfg.lr_init)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5,
                patience=1, verbose=False, threshold=0.0001, threshold_mode='abs', cooldown=0, min_lr=1e-8, eps=1e-08)

        self._check_dirs()
        self._load_data("train")
        self._load_ckpts()
        self.writer = SummaryWriter(log_dir=self.cfg.log_dir)
 
    def train(self):
        self.net.train()
        iteration = self.step
        for epch in range(self.start_epoch, self.cfg.epochs):
            running_loss = 0.
            epch_loss = 0.
            
            for idx, batch in enumerate(self.train_data, start=0):
                inputs, labels = batch[0].to(self.device), batch[1].to(self.device)
                self.optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                epch_loss += loss.item()

                if idx % self.cfg.verbose_step == self.cfg.verbose_step - 1:
                    valid_acc, valid_loss = self._validation()
                    self.writer.add_scalar("Loss/Train", running_loss/self.cfg.verbose_step, iteration)
                    self.writer.add_scalar("Loss/Validation", valid_loss, iteration)
                    self.writer.add_scalar("Acc/Validation", valid_acc, iteration)
                    self.writer.add_scalar("LearningRate", self.optimizer.param_groups[0]["lr"], iteration)
                    ckpt_name = f"step_{iteration}_val_loss_{valid_loss:.3}_{valid_acc:.2f}.pth"
                    self._save_ckpts(iteration, epch, ckpt_name) 

                    print(f"[{epch}/{iteration}] Loss/train: {running_loss/self.cfg.verbose_step:.3}, Loss/val: {valid_loss:.3}, Acc/val: {valid_acc:7.2f}, lr: {self.optimizer.param_groups[0]['lr']:.3}")
                    running_loss = 0.
                    iteration += 1

            self.scheduler.step(metrics=valid_loss)

    def test(self):
        self.net.eval()
        self._load_data("test")

        correct = 0
        total = 0

        with torch.no_grad():
            for batch in self.test_data:
                inputs, labels = batch[0].to(self.device), batch[1].to(self.device)
                outputs = self.net(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc = correct / total * 100
        print(f"[*] Test accuracy: {acc:.2f}%")

        self.net.train()

    def _validation(self):
        self.net.eval()
        correct = 0
        total = 0
        total_loss = 0
        with torch.no_grad():
            for data in self.valid_data:
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                total_loss += loss
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        self.net.train()
        acc = correct / total * 100
        batch_loss = total_loss / len(self.valid_data)
        return acc, batch_loss


    def _save_ckpts(self, step, epoch, filename):
        state = {"network": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "step": step,
                "epoch": epoch}
        torch.save(state, os.path.join(self.cfg.ckpts_dir, filename))


    def _load_ckpts(self):
        self.step = 1
        self.start_epoch = 0
        if os.path.exists(self.cfg.resume):
            resume_path = self.cfg.resume
        else:
            ckpts = [[f, int(f.split("_")[1])] for f in os.listdir(self.cfg.ckpts_dir) if f.endswith(".pth")]
            ckpts.sort(key = lambda x: x[1], reverse=True)
            resume_path = None if len(ckpts) == 0 else os.path.join(self.cfg.ckpts_dir, ckpts[0][0])

        if resume_path and os.path.exists(resume_path):
            print(f"[*] CKPTs Loading {resume_path}")
            checkpoint = torch.load(resume_path, map_location="cpu")
            self.step = checkpoint["step"] + 1
            self.net.load_state_dict(checkpoint["network"])

            if not self.cfg.optimizer_reset:
                self.optimizer.load_state_dict(checkpoint["optimizer"])

            if "epoch" in checkpoint:
                self.start_epoch = checkpoint["epoch"]

            print(f"[*] CKPTs Loaded {resume_path} (Continue from epoch {self.start_epoch} and step {self.step})")
            del checkpoint
            torch.cuda.empty_cache()
        else:
            print(f"[*] No CKPTs found at {resume_path}")

    def _check_dirs(self):
        if not os.path.exists(self.cfg.log_dir):
            os.mkdir(self.cfg.log_dir)
        if not os.path.exists(self.cfg.ckpts_dir):
            os.mkdir(self.cfg.ckpts_dir)

    def _load_data(self, *args):
        if args[0] == "train":
            data = self.dataset(root = self.data_dir, 
                    train=True, download=True, transform=self.train_transforms)
            data_size = len(data)
            indices = list(range(data_size))
            valid_ratio = 0.1
            split = int(np.floor(valid_ratio*data_size))
            train_idx, valid_idx = indices[split:], indices[:split]
            train_sampler = SubsetRandomSampler(train_idx)
            valid_sampler = SubsetRandomSampler(valid_idx)

            self.train_data = torch.utils.data.DataLoader(data,
                    batch_size=self.cfg.batch_size,
                    sampler=train_sampler,
                    num_workers=1)

            self.valid_data = torch.utils.data.DataLoader(data,
                    batch_size=self.cfg.batch_size,
                    sampler=train_sampler,
                    num_workers=1)
        elif args[0] == "test":
            data = self.dataset(root=self.data_dir,
                    train=False, download=True, transform=self.test_transforms)
            self.test_data = torch.utils.data.DataLoader(data,
                    batch_size=self.cfg.batch_size,
                    shuffle=False,
                    num_workers=1)

@hydra.main(config_path="./default.yaml")
def main(cfg):
    app=Application(cfg.parameters)
    app.build()
    if not cfg.parameters.inference:
        app.train()
    app.test()


if __name__=="__main__":
    main()
