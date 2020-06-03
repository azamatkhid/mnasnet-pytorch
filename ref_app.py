import torch
import torch.nn as nn
from torchvision import models
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
import hydra

from app import Application

class MnasNet_official(Application):
    def __init__(self, cfg):
        super(MnasNet_official, self).__init__(cfg)
        pass

    def build(self, criterion=nn.CrossEntropyLoss, optimizer=torch.optim.RMSprop):
        model = getattr(models, "mnasnet1_0")
        self.net = model(pretrained = False, num_classes = self.cfg.num_classes)
        self.net = torch.nn.DataParallel(self.net)
        
        if torch.cuda.device_count() > 1:
            print(f"[*] Number of GPUs: {torch.cuda.device_count()}")

        self.net.to(self.device)
        if self.cfg.verbose and torch.cuda.device_count() <= 1:
            summary(self.net,(3,224,224))
        
        self.criterion = criterion()
        self.optimizer = optimizer(self.net.parameters(), lr=self.cfg.lr_init)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5,
                patience=1, verbose=False, threshold=0.0001, threshold_mode='abs', cooldown=0, min_lr=1e-8, eps=1e-08)
        
        self._check_dirs()
        self._load_data("train")
        self._load_ckpts()
        self.writer = SummaryWriter(log_dir=self.cfg.log_dir)


@hydra.main(config_path="./default.yaml")
def main(cfg):
    app=MnasNet_official(cfg.parameters)
    app.build()
    if not cfg.parameters.inference:
        app.train()
    app.test()

if __name__=="__main__":
    main()
