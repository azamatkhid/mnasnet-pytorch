import torch
import torch.nn as nn
from torchsummary import summary

class SepConv(nn.Module):
    def __init__(self,inchannel,outchannel):
        super(SepConv,self).__init__()
        self.dwconv = nn.Conv2d(inchannel, inchannel, kernel_size = 3, stride = 1, bias = False)
        self.bn0 = nn.BatchNorm2d(inchannel)
        self.act0 = nn.ReLU(inplace = True)
        self.conv = nn.Conv2d(inchannel, outchannel, kernel_size = 1, stride = 0, bias = False)
        self.bn1 = nn.BatchNorm2d(outchannel)
        self.act1 = nn.ReLU(inplace = True)

    def forward(self, x):
        out = self.dwconv(x)
        out = self.act0(self.bn0(out))
        out = self.conv(out)
        out = self.act1(self.bn1(out))
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
    sep_conv=SepConv(3,3)
    print(sep_conv)
    summary(sep_conv,(3,224,224))

if __name__=="__main__":
    main()
