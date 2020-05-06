import torch
from torchvision import models
from torchsummary import summary
import hydra

from app import Application

class MnasNet_official(Application):
    def __init__(self, cfg):
        super(MnasNet_official, self).__init__(cfg)
        pass

    def build(self):
        if self.cfg.model=="mnasnet-b1":
            model = getattr(models, "mnasnet1_0")
        self.net = model(pretrained = False, num_classes = self.cfg.num_classes)

        if torch.cuda.device_count() > 1:
            self.net = nn.DataParallel(self.net)
            print(f"Number of GPUs: {torch.cuda.device_count()}")

        self.net.to(self.device)
        if self.cfg.verbose and torch.cuda.device_count() <= 1:
            summary(self.net,(3,224,224))


@hydra.main(config_path="./default.yaml")
def main(cfg):
    app=MnasNet_official(cfg.parameters)
    app.build()
    app.train()
    app.test()

if __name__=="__main__":
    main()
