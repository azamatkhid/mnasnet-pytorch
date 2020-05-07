import torch
import torch.nn as nn
from torchsummary import summary

class _InvertedResidual(nn.Module):
    def __init__(self, inchannel, outchannel, kernel_size, stride=1, exp_size=1, se=False, se_ratio=0.25):
        super(_InvertedResidual, self).__init__()
        self.se = se
        self.skip = inchannel == outchannel and stride == 1
        
        self.conv0 = nn.Conv2d(inchannel, exp_size*inchannel, kernel_size=1, stride=1, bias=False)
        self.bn0 = nn.BatchNorm2d(exp_size*inchannel)
        self.act0 = nn.ReLU(inplace = True)

        self.conv1 = nn.Conv2d(exp_size*inchannel, exp_size*inchannel, 
                kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2,
                bias=False, groups=exp_size*inchannel)
        self.bn1 = nn.BatchNorm2d(exp_size*inchannel)
        self.act1 = nn.ReLU(inplace=True)

        self.squeeze = nn.Sequential()

        if self.se:
            self.squeeze = nn.Sequential(
                    nn.AdaptiveAvgPool2d((1,1)),
                    nn.Conv2d(exp_size*inchannel, int(se_ratio*exp_size*inchannel), 1, padding=0, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(int(se_ratio*exp_size*inchannel), exp_size*inchannel, 1, padding=0, bias=True),
                    nn.Sigmoid())

        self.conv2 = nn.Conv2d(exp_size*inchannel , outchannel, kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outchannel)

    def forward(self,x):
        out = self.act0(self.bn0(self.conv0(x)))
        out = self.act1(self.bn1(self.conv1(out)))
        
        if self.se:
            out = out * self.squeeze(out)

        out = self.bn2(self.conv2(out))

        if self.skip:
            out = out + x
        return out

class MnasNet(nn.Module):
    def __init__(self, inchannel, cfg, training=True):
        super(MnasNet,self).__init__()
        self.p = cfg.dropout
        self.training = training

        if cfg.model == "mnasnet-a1":
            self._mnasnet_a1(inchannel)
            self.model = nn.Sequential(*self.blocks)
            self.logits = nn.Linear(320, cfg.num_classes)
        elif cfg.model == "mnasnet-b1":
            self._mnasnet_b1(inchannel)
            self.model = nn.Sequential(*self.blocks)
            self.logits = nn.Linear(1280, cfg.num_classes)
        
        pass

    def forward(self,x):
        out = nn.functional.adaptive_avg_pool2d(self.model(x),(1,1))
        out = self.logits(nn.functional.dropout(torch.flatten(out,1), p=self.p, training=self.training))
        return out

    def _mnasnet_a1(self, inchannel):
        self.blocks = []

        self.blocks.append(nn.Conv2d(inchannel, 32, kernel_size=3, stride=2, padding=1, bias=False))
        self.blocks.append(nn.BatchNorm2d(32))
        self.blocks.append(nn.ReLU(inplace=True))
        
        self.blocks.append(nn.Conv2d(32, 32, kernel_size=3, stride=1, 
                padding=1, bias=False, groups=32))
        self.blocks.append(nn.BatchNorm2d(32))
        self.blocks.append(nn.ReLU(inplace=True))
        self.blocks.append(nn.Conv2d(32, 16, kernel_size=1, stride=1,
                padding=0, bias=False))
        self.blocks.append(nn.BatchNorm2d(16))

        self.blocks.append(_InvertedResidual(16, 24, 3, stride=2, exp_size=6, se=False))
        self.blocks.append(_InvertedResidual(24, 24, 3, stride=1, exp_size=6, se=False))

        self.blocks.append(_InvertedResidual(24, 40, 5, stride=2, exp_size=3, se=True))
        self.blocks.append(_InvertedResidual(40, 40, 5, stride=1, exp_size=3, se=True))
        self.blocks.append(_InvertedResidual(40, 40, 5, stride=1, exp_size=3, se=True))

        self.blocks.append(_InvertedResidual(40, 80, 3, stride=2, exp_size=6, se=False))
        self.blocks.append(_InvertedResidual(80, 80, 3, stride=1, exp_size=6, se=False))
        self.blocks.append(_InvertedResidual(80, 80, 3, stride=1, exp_size=6, se=False))
        self.blocks.append(_InvertedResidual(80, 80, 3, stride=1, exp_size=6, se=False))

        self.blocks.append(_InvertedResidual(80, 112, 3, stride=1, exp_size=6, se=True))
        self.blocks.append(_InvertedResidual(112, 112, 3, stride=1, exp_size=6, se=True))

        self.blocks.append(_InvertedResidual(112, 160, 5, stride=2, exp_size=6, se=True))
        self.blocks.append(_InvertedResidual(160, 160, 5, stride=1, exp_size=6, se=True))
        self.blocks.append(_InvertedResidual(160, 160, 5, stride=1, exp_size=6, se=True))

        self.blocks.append(_InvertedResidual(160, 320, 3, stride=1, exp_size=6, se=False))
        pass

    def _mnasnet_b1(self, inchannel):
        self.blocks = []

        self.blocks.append(nn.Conv2d(inchannel, 32, kernel_size=3, stride=2, padding=1, bias=False))
        self.blocks.append(nn.BatchNorm2d(32))
        self.blocks.append(nn.ReLU(inplace=True))
        
        self.blocks.append(nn.Conv2d(32, 32, kernel_size=3, stride=1, 
                padding=1, bias=False, groups=32))
        self.blocks.append(nn.BatchNorm2d(32))
        self.blocks.append(nn.ReLU(inplace=True))
        self.blocks.append(nn.Conv2d(32, 16, kernel_size=1, stride=1,
                padding=0, bias=False))
        self.blocks.append(nn.BatchNorm2d(16))

        self.blocks.append(_InvertedResidual(16, 24, 3, stride=2, exp_size=3, se=False))
        self.blocks.append(_InvertedResidual(24, 24, 3, stride=1, exp_size=3, se=False))
        self.blocks.append(_InvertedResidual(24, 24, 3, stride=1, exp_size=3, se=False))

        self.blocks.append(_InvertedResidual(24, 40, 5, stride=2, exp_size=3, se=False))
        self.blocks.append(_InvertedResidual(40, 40, 5, stride=1, exp_size=3, se=False))
        self.blocks.append(_InvertedResidual(40, 40, 5, stride=1, exp_size=3, se=False))

        self.blocks.append(_InvertedResidual(40, 80, 5, stride=2, exp_size=6, se=False))
        self.blocks.append(_InvertedResidual(80, 80, 5, stride=1, exp_size=6, se=False))
        self.blocks.append(_InvertedResidual(80, 80, 5, stride=1, exp_size=6, se=False))

        self.blocks.append(_InvertedResidual(80, 96, 3, stride=1, exp_size=6, se=False))
        self.blocks.append(_InvertedResidual(96, 96, 3, stride=1, exp_size=6, se=False))

        self.blocks.append(_InvertedResidual(96, 192, 5, stride=2, exp_size=6, se=False))
        self.blocks.append(_InvertedResidual(192, 192, 5, stride=1, exp_size=6, se=False))
        self.blocks.append(_InvertedResidual(192, 192, 5, stride=1, exp_size=6, se=False))
        self.blocks.append(_InvertedResidual(192, 192, 5, stride=1, exp_size=6, se=False))

        self.blocks.append(_InvertedResidual(192, 320, 3, stride=1, exp_size=6, se=False))
        self.blocks.append(nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False))
        self.blocks.append(nn.BatchNorm2d(1280))
        self.blocks.append(nn.ReLU(inplace=True))
        pass


def main():
    sep_conv=MnasNet(3)
    summary(sep_conv, (3, 224, 224))

if __name__=="__main__":
    main()
