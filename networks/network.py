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

class conv_block(nn.Module):
    def __init__(self,ch_in : int,ch_out : int, dropout : float= 0.) -> None:
        super(conv_block,self).__init__()
        layers = [
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        ]
        if dropout != 0.:
            layers.append(nn.Dropout(p=dropout))
        self.conv = nn.Sequential(*layers)



    def forward(self,x : torch.Tensor)-> torch.Tensor:
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in: int,ch_out: int, dropout: float= 0.) -> None:
        super(up_conv,self).__init__()
        layers = [
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        ]
        if dropout != 0.:
            layers.append(nn.Dropout(p=dropout))
        self.up = nn.Sequential(*layers)


    def forward(self,x : torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        return x

class Recurrent_block(nn.Module):
    def __init__(self,ch_out : int,t: int = 2):
        super(Recurrent_block,self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x : torch.Tensor) -> torch.Tensor:
        for i in range(self.t):

            if i==0:
                x1 = self.conv(x)
            
            x1 = self.conv(x+x1)
        return x1
        
class RRCNN_block(nn.Module):
    def __init__(self, ch_in : int, ch_out : int, t: int = 2):
        super(RRCNN_block,self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out,t=t),
            Recurrent_block(ch_out,t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=1,padding=0)

    def forward(self,x : torch.Tensor) -> torch.Tensor:
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x+x1


class single_conv(nn.Module):
    def __init__(self,ch_in: int, ch_out: int) -> None:
        super(single_conv,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x : torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return x

class Attention_block(nn.Module):
    def __init__(self, F_g: int, F_l: int, F_int: int) -> None:
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi


class U_Net(nn.Module):
    def __init__(self,img_ch: int = 3,output_ch: int = 1, dropout: float = 0.) -> None:
        super(U_Net,self).__init__()
        self.output_ch = output_ch
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in= img_ch, ch_out= 64, dropout= dropout)
        self.Conv2 = conv_block(ch_in= 64, ch_out= 128, dropout= dropout)
        self.Conv3 = conv_block(ch_in= 128, ch_out= 256, dropout= dropout)
        self.Conv4 = conv_block(ch_in= 256, ch_out= 512, dropout= dropout)
        self.Conv5 = conv_block(ch_in= 512, ch_out= 1024, dropout= dropout)

        self.Up5 = up_conv(ch_in= 1024, ch_out= 512)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512,dropout= dropout)

        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256,dropout= dropout)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128,dropout= dropout)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64,dropout= dropout)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x: torch.Tensor) -> torch.Tensor:
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)
        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        return d1


class R2U_Net(nn.Module):
    def __init__(self,img_ch: int= 3, output_ch: int= 1,t: int= 2) -> None:
        super(R2U_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_ch,ch_out=64,t=t)

        self.RRCNN2 = RRCNN_block(ch_in=64,ch_out=128,t=t)
        
        self.RRCNN3 = RRCNN_block(ch_in=128,ch_out=256,t=t)
        
        self.RRCNN4 = RRCNN_block(ch_in=256,ch_out=512,t=t)
        
        self.RRCNN5 = RRCNN_block(ch_in=512,ch_out=1024,t=t)
        

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512,t=t)
        
        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256,t=t)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128,t=t)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64,t=t)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x: torch.Tensor) -> torch.Tensor:
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.Up_RRCNN5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)

        return d1



class AttU_Net(nn.Module):
    def __init__(self,img_ch: int=3, output_ch: int= 1, dropout: float = 0.)-> None:
        super(AttU_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=64, dropout= dropout)
        self.Conv2 = conv_block(ch_in=64,ch_out=128, dropout= dropout)
        self.Conv3 = conv_block(ch_in=128,ch_out=256, dropout= dropout)
        self.Conv4 = conv_block(ch_in=256,ch_out=512, dropout= dropout)
        self.Conv5 = conv_block(ch_in=512,ch_out=1024, dropout= dropout)

        self.Up5 = up_conv(ch_in=1024,ch_out=512, dropout=dropout)
        self.Att5 = Attention_block(F_g=512,F_l=512,F_int=256)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512, dropout=dropout)

        self.Up4 = up_conv(ch_in=512,ch_out=256, dropout=dropout)
        self.Att4 = Attention_block(F_g=256,F_l=256,F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256, dropout=dropout)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128, dropout=dropout)
        self.Att3 = Attention_block(F_g=128,F_l=128,F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128, dropout=dropout)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64, dropout=dropout)
        self.Att2 = Attention_block(F_g=64,F_l=64,F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64, dropout=dropout)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x: torch.Tensor) -> torch.Tensor:
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1

class slave_conv_block(nn.Module):
    def __init__(self,ch_in : int,ch_out : int, dropout : float= 0.) -> None:
        super(slave_conv_block,self).__init__()
        layers = [
            nn.Conv2d(ch_in, ch_out, kernel_size=1,stride=1,padding=0,),
            nn.LayerNorm(ch_out),
            nn.ReLU(inplace=True),
        ]
        if dropout != 0.:
            layers.append(nn.Dropout(p=dropout))
        self.conv = nn.Sequential(*layers)



    def forward(self,x : torch.Tensor)-> torch.Tensor:
        x = self.conv(x)
        return x

class SlaveAttU_Net(nn.Module):
    def __init__(self,img_ch: int=3, output_ch: int= 1, dropout: float = 0.)-> None:
        super(SlaveAttU_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=64, dropout= dropout)
        self.Conv2 = conv_block(ch_in=64,ch_out=128, dropout= dropout)
        self.Conv3 = conv_block(ch_in=128,ch_out=256, dropout= dropout)
        self.Conv4 = conv_block(ch_in=256,ch_out=512, dropout= dropout)
        self.Conv5 = conv_block(ch_in=512,ch_out=1024, dropout= dropout)

        self.SlaveConv5 = slave_conv_block(ch_in=2048,ch_out=1024, dropout= dropout)
        self.SlaveConv4 = slave_conv_block(ch_in= 1024,ch_out=512, dropout= dropout)
        self.SlaveConv3 = slave_conv_block(ch_in= 512, ch_out= 256, dropout= dropout)
        self.SlaveConv2 = slave_conv_block(ch_in= 256,ch_out= 128, dropout= dropout)
        self.SlaveConv1 = slave_conv_block(ch_in= 128,ch_out= 64, dropout= dropout)


        self.Up5 = up_conv(ch_in=1024,ch_out=512, dropout=dropout)
        self.Att5 = Attention_block(F_g=512,F_l=512,F_int=256)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512, dropout=dropout)

        self.Up4 = up_conv(ch_in=512,ch_out=256, dropout=dropout)
        self.Att4 = Attention_block(F_g=256,F_l=256,F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256, dropout=dropout)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128, dropout=dropout)
        self.Att3 = Attention_block(F_g=128,F_l=128,F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128, dropout=dropout)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64, dropout=dropout)
        self.Att2 = Attention_block(F_g=64,F_l=64,F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64, dropout=dropout)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)

    def forward(self,x: torch.Tensor, sl: torch.Tensor) -> torch.Tensor:
        # encoding path
        x1 = self.Conv1(x)
        sl1 = self.Conv1(sl)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        sl2 = self.Maxpool(sl1)
        sl2 = self.Conv2(sl2)
        
        sl3 = self.Maxpool(sl2)
        sl3 = self.Conv3(sl3)

        sl4 = self.Maxpool(sl3)
        sl4 = self.Conv4(sl4)

        sl5 = self.Maxpool(sl4)
        sl5 = self.Conv5(sl5)

        # print(f"\nX1 size : {x1.shape}")
        x1 = torch.cat((x1,sl1),dim=1) 
        # print(f"\n [CAT] X1 size : {x1.shape}")

        x2 = torch.cat((x2,sl2),dim=1) 
        x3 = torch.cat((x3,sl3),dim=1) 
        x4 = torch.cat((x4,sl4),dim=1) 
        x5 = torch.cat((x5,sl5),dim=1) 
        
        x1= self.SlaveConv1(x1)
        # print(f"X1 size : {x1.shape}")

        x2= self.SlaveConv2(x2)
        # print(f"x2 size : {x2.shape}")

        x3= self.SlaveConv3(x3)
        # print(f"x3 size : {x3.shape}")

        x4= self.SlaveConv4(x4)
        # print(f"x4 size : {x4.shape}")

        x5= self.SlaveConv5(x5)
        # print(f"x5 size : {x5.shape}")

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.Up_conv5(d5)
        # print(f"d5 size : {d5.shape}")        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)
        # print(f"d4 size : {d4.shape}")
        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)
        # print(f"d3 size : {d3.shape}")
        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)
        # print(f"d2 size : {d2.shape}")
        d1 = self.Conv_1x1(d2)
        # print(f"d1 size : {d1.shape}")
        return d1


class R2AttU_Net(nn.Module):
    def __init__(self,img_ch: int= 3, output_ch: int= 1, t: int= 2)-> None:
        super(R2AttU_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_ch,ch_out=64,t=t)

        self.RRCNN2 = RRCNN_block(ch_in=64,ch_out=128,t=t)
        
        self.RRCNN3 = RRCNN_block(ch_in=128,ch_out=256,t=t)
        
        self.RRCNN4 = RRCNN_block(ch_in=256,ch_out=512,t=t)
        
        self.RRCNN5 = RRCNN_block(ch_in=512,ch_out=1024,t=t)
        

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Att5 = Attention_block(F_g=512,F_l=512,F_int=256)
        self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512,t=t)
        
        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Att4 = Attention_block(F_g=256,F_l=256,F_int=128)
        self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256,t=t)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Att3 = Attention_block(F_g=128,F_l=128,F_int=64)
        self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128,t=t)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Att2 = Attention_block(F_g=64,F_l=64,F_int=32)
        self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64,t=t)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x: torch.Tensor) -> torch.Tensor:
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.Up_RRCNN5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)

        return d1
