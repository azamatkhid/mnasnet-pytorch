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
        self.stride = stride
        
        self.conv0 = nn.Conv2d(inchannel, exp_size * inchannel, kernel_size = 1, stride = 1, bias = False)
        self.bn0 = nn.BatchNorm2d(exp_size * inchannel)
        self.act0 = nn.ReLU(inplace = True)

        self.conv1 = nn.Conv2d(exp_size * inchannel, exp_size * inchannel, 
                kernel_size = kernel_size, stride = self.stride, padding = (kernel_size - 1) // 2,
                bias = False, groups = exp_size * inchannel)
        self.bn1 = nn.BatchNorm2d(exp_size * inchannel)
        self.act1 = nn.ReLU(inplace = True)

        self.squeeze = nn.Sequential()

        if self.se:
            self.pooling = nn.AdaptiveAvgPool2d((1,1))
            self.squeeze = nn.Sequential(
                    nn.AdaptiveAvgPool2d((1,1)),
                    nn.Conv2d(exp_size * inchannel, int(se_ratio * exp_size * inchannel), 1, padding=0, bias=False),
                    nn.ReLU(inplace = True),
                    nn.Conv2d(int(se_ratio * exp_size * inchannel), exp_size * inchannel, 1, padding=0, bias=False),
                    nn.Sigmoid())

        self.conv2 = nn.Conv2d(exp_size* inchannel , outchannel, kernel_size = 1, stride = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(outchannel)

    def forward(self,x):
        out = self.act0(self.bn0(self.conv0(x)))
        out = self.act1(self.bn1(self.conv1(out)))
        if self.se:
            out = out * self.squeeze(out)
        out = self.bn2(self.conv2(out))
        if self.stride==1:
            out = out + x
        return out

#class MnasNet(nn.Module):
#    def __init__(self,**kwargs):
#        super(MnasNet,self).__init__()
#        pass 
#
#    def forward(self,x):
#        out=x
#        return out

def main():
    sep_conv=InvertedBottleNeck(16, 16, 5, stride = 2, exp_size = 3, se = True)
#    print(sep_conv)
    summary(sep_conv, (16,112,112))

if __name__=="__main__":
    main()
