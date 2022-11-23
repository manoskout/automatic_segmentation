import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torchio as tio

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

def init_normalization(norm_type, input_ch):
    if norm_type == "batch":
        # print(f"Selected norm: {norm_type}")
        return nn.BatchNorm2d(input_ch)
    elif norm_type == "instance":
        print(f"Selected norm: {norm_type}")
        return nn.InstanceNorm2d(input_ch)
    elif norm_type == "group":
        # TODO -> Test also with bigger number of group
        return nn.GroupNorm(num_groups= 8, num_channels=input_ch)


class conv_block(nn.Module):
    def __init__(self,ch_in : int,ch_out : int, dropout : float= 0., normalization: str = "batch" ) -> None:
        super(conv_block,self).__init__()
        layers = [
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            init_normalization(normalization, ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            init_normalization(normalization, ch_out),
            nn.ReLU(inplace=True)
        ]
        if dropout != 0.:
            layers.append(nn.Dropout(p=dropout))
        self.conv = nn.Sequential(*layers)



    def forward(self,x : torch.Tensor)-> torch.Tensor:
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in: int,ch_out: int, dropout: float= 0., normalization: str = "batch") -> None:
        super(up_conv,self).__init__()
        layers = [
            nn.Upsample(mode="bilinear",scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    init_normalization(normalization, ch_out),
			nn.ReLU(inplace=True)
        ]
        if dropout != 0.:
            layers.append(nn.Dropout(p=dropout))
        self.up = nn.Sequential(*layers)


    def forward(self,x : torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        return x

class Recurrent_block(nn.Module):
    def __init__(self,ch_out : int,t: int = 2, normalization: str = "batch"):
        super(Recurrent_block,self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    init_normalization(normalization, ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x : torch.Tensor) -> torch.Tensor:
        for i in range(self.t):

            if i==0:
                x1 = self.conv(x)
            
            x1 = self.conv(x+x1)
        return x1

class Residual_block(nn.Module):
    def __init__(self,ch_in: int, ch_out : int,t: int = 2,dropout: float = 0., normalization: str = "batch"):
        super(Residual_block,self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv1 = nn.Sequential(
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    init_normalization(normalization, ch_out),
			nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    init_normalization(normalization, ch_out),
			nn.ReLU(inplace=True)
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self,x : torch.Tensor) -> torch.Tensor:

        for i in range(self.t):
            if i==0:
                x = self.conv1(x)
            x = self.dropout(x)
            x1 = self.conv2(x)
        skip = x+x1
        return skip
        
class RRCNN_block(nn.Module):
    def __init__(self, ch_in : int, ch_out : int, t: int = 2, normalization: str= "batch"):
        super(RRCNN_block,self).__init__()
        self.RCNN = nn.Sequential(
            init_normalization(normalization, ch_out),
            init_normalization(normalization, ch_out),

        )
        self.Conv_1x1 = nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=1,padding=0)

    def forward(self,x : torch.Tensor) -> torch.Tensor:
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x+x1



class Attention_block(nn.Module):
    def __init__(self, F_g: int, F_l: int, F_int: int, normalization: str = "batch") -> None:
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            init_normalization(normalization, F_int),
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            init_normalization(normalization, F_int),
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            # nn.Softmax()
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi