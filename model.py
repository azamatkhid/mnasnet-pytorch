import torch
import torch.nn as nn
from torchsummary import summary

class SepConv(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(SepConv, self).__init__()
        self.conv0 = nn.Conv2d(inchannel, inchannel, kernel_size = 3, stride = 1, 
                padding = 1, bias = False, groups = inchannel)
        self.bn0 = nn.BatchNorm2d(inchannel)
        self.act0 = nn.ReLU(inplace = True)

        self.conv1 = nn.Conv2d(inchannel, outchannel, kernel_size = 1, stride = 1,
                padding = 0, bias = False)
        self.bn1 = nn.BatchNorm2d(outchannel)

    def forward(self, x):
        out = self.conv0(x)
        out = self.act0(self.bn0(out))
        out = self.conv1(out)
        out = self.bn1(out)
        return out

class InvertedBottleNeck(nn.Module):
    def __init__(self, inchannel, outchannel, kernel_size, stride = 1, exp_size = 1, se = False, se_ratio = 0.25):
        super(InvertedBottleNeck, self).__init__()
        self.se = se
        self.skip = inchannel == outchannel and stride == 1
        
        self.conv0 = nn.Conv2d(inchannel, exp_size * inchannel, kernel_size = 1, stride = 1, bias = False)
        self.bn0 = nn.BatchNorm2d(exp_size * inchannel)
        self.act0 = nn.ReLU(inplace = True)

        self.conv1 = nn.Conv2d(exp_size * inchannel, exp_size * inchannel, 
                kernel_size = kernel_size, stride = stride, padding = (kernel_size - 1) // 2,
                bias = False, groups = exp_size * inchannel)
        self.bn1 = nn.BatchNorm2d(exp_size * inchannel)
        self.act1 = nn.ReLU(inplace = True)

        self.squeeze = nn.Sequential()

        if self.se:
            self.squeeze = nn.Sequential(
                    nn.AdaptiveAvgPool2d((1,1)),
                    nn.Conv2d(exp_size * inchannel, int(se_ratio * exp_size * inchannel), 1, padding = 0, bias = True),
                    nn.ReLU(inplace = True),
                    nn.Conv2d(int(se_ratio * exp_size * inchannel), exp_size * inchannel, 1, padding = 0, bias = True),
                    nn.Sigmoid())

        self.conv2 = nn.Conv2d(exp_size* inchannel , outchannel, kernel_size = 1, stride = 1, bias = False)
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
    def __init__(self,inchannel, num_classes = 1000):
        super(MnasNet,self).__init__()

        self.block0 = nn.Sequential(
                nn.Conv2d(inchannel, 32, kernel_size = 3, stride = 2, padding = 1, bias = False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace = True))
        
        self.block1 = SepConv(32, 16)
        
       
        self.block2 = nn.Sequential(
                InvertedBottleNeck(16, 24, 3, stride = 2, exp_size = 6, se = False),
                InvertedBottleNeck(24, 24, 3, stride = 1, exp_size = 6, se = False))
        
        self.block3 = nn.Sequential(
                InvertedBottleNeck(24, 40, 5, stride = 2, exp_size = 3, se = True),
                InvertedBottleNeck(40, 40, 5, stride = 1, exp_size = 3, se = True),
                InvertedBottleNeck(40, 40, 5, stride = 1, exp_size = 3, se = True))

        self.block4 = nn.Sequential(
                InvertedBottleNeck(40, 80, 3, stride = 2, exp_size = 6, se = False),
                InvertedBottleNeck(80, 80, 3, stride = 1, exp_size = 6, se = False),
                InvertedBottleNeck(80, 80, 3, stride = 1, exp_size = 6, se = False),
                InvertedBottleNeck(80, 80, 3, stride = 1, exp_size = 6, se = False))
 
        self.block5 = nn.Sequential(
                InvertedBottleNeck(80, 112, 3, stride = 1, exp_size = 6, se = True),
                InvertedBottleNeck(112, 112, 3, stride = 1, exp_size = 6, se = True))
        
        self.block6 = nn.Sequential(
                InvertedBottleNeck(112, 160, 5, stride = 2, exp_size = 6, se = True),
                InvertedBottleNeck(160, 160, 5, stride = 1, exp_size = 6, se = True),
                InvertedBottleNeck(160, 160, 5, stride = 1, exp_size = 6, se = True))

        self.block7 = InvertedBottleNeck(160, 320, 3, stride = 1, exp_size = 6, se = False)

        self.pooling = nn.AdaptiveAvgPool2d((1,1))
        self.logits = nn.Linear(320, num_classes)
        pass 

    def forward(self,x):
        out = self.block0(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.block6(out)
        out = self.block7(out)
        out = self.pooling(out)
        out = self.logits(torch.flatten(out,1))
        return out

def main():
    sep_conv=MnasNet(3)
#    print(sep_conv)
    summary(sep_conv, (3,224,224))

if __name__=="__main__":
    main()
