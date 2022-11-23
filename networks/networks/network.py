from cv2 import norm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torchio as tio
from networks.parts import conv_block, up_conv, Attention_block, Residual_block, RRCNN_block, init_weights

class U_Net(nn.Module):
    def __init__(self,img_ch: int = 3,output_ch: int = 1, dropout: float = 0., normalization: str = "batch") -> None:
        super(U_Net,self).__init__()
        self.output_ch = output_ch
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in= img_ch, ch_out= 64, dropout= dropout, normalization=normalization)
        self.Conv2 = conv_block(ch_in= 64, ch_out= 128, dropout= dropout, normalization=normalization)
        self.Conv3 = conv_block(ch_in= 128, ch_out= 256, dropout= dropout, normalization=normalization)
        self.Conv4 = conv_block(ch_in= 256, ch_out= 512, dropout= dropout, normalization=normalization)
        self.Conv5 = conv_block(ch_in= 512, ch_out= 1024, dropout= dropout, normalization=normalization)

        self.Up5 = up_conv(ch_in= 1024, ch_out= 512, normalization=normalization)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512,dropout= dropout, normalization=normalization)

        self.Up4 = up_conv(ch_in=512,ch_out=256, normalization=normalization)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256,dropout= dropout, normalization=normalization)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128, normalization=normalization)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128,dropout= dropout, normalization=normalization)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64, normalization=normalization)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64,dropout= dropout, normalization=normalization)

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
    def __init__(self,img_ch: int=3, output_ch: int= 1, dropout: float = 0., normalization: str = "batch")-> None:
        super(AttU_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=64, dropout= dropout, normalization=normalization)
        self.Conv2 = conv_block(ch_in=64,ch_out=128, dropout= dropout, normalization=normalization)
        self.Conv3 = conv_block(ch_in=128,ch_out=256, dropout= dropout, normalization=normalization)
        self.Conv4 = conv_block(ch_in=256,ch_out=512, dropout= dropout, normalization=normalization)
        self.Conv5 = conv_block(ch_in=512,ch_out=1024, dropout= dropout, normalization=normalization)

        self.Up5 = up_conv(ch_in=1024,ch_out=512, dropout=dropout, normalization=normalization)
        self.Att5 = Attention_block(F_g=512,F_l=512,F_int=256, normalization=normalization)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512, dropout=dropout, normalization=normalization)

        self.Up4 = up_conv(ch_in=512,ch_out=256, dropout=dropout, normalization=normalization)
        self.Att4 = Attention_block(F_g=256,F_l=256,F_int=128, normalization=normalization)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256, dropout=dropout, normalization=normalization)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128, dropout=dropout, normalization=normalization)
        self.Att3 = Attention_block(F_g=128,F_l=128,F_int=64, normalization=normalization)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128, dropout=dropout, normalization=normalization)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64, dropout=dropout, normalization=normalization)
        self.Att2 = Attention_block(F_g=64,F_l=64,F_int=32, normalization=normalization)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64, dropout=dropout, normalization=normalization)

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

class ResAttU_Net(nn.Module):
    def __init__(self,img_ch: int= 3, output_ch: int= 1, t: int= 2, dropout: float = 0., normalization: str= "batch")-> None:
        super(ResAttU_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.Upsample = nn.Upsample(mode="bilinear",scale_factor=2)

        self.Res1 = Residual_block(ch_in=img_ch,ch_out=64,t=t, dropout=dropout, normalization=normalization)

        self.Res2 = Residual_block(ch_in=64,ch_out=128,t=t, dropout=dropout, normalization=normalization)
        
        self.Res3 = Residual_block(ch_in=128,ch_out=256,t=t, dropout=dropout, normalization=normalization)
        
        self.Res4 = Residual_block(ch_in=256,ch_out=512,t=t, dropout=dropout, normalization=normalization)
        
        self.Res5 = Residual_block(ch_in=512,ch_out=1024,t=t, dropout=dropout, normalization=normalization)
        

        self.Up5 = up_conv(ch_in=1024,ch_out=512, dropout=dropout, normalization=normalization)
        self.Att5 = Attention_block(F_g=512,F_l=512,F_int=256, normalization=normalization)
        self.Up_Res5 = Residual_block(ch_in=1024, ch_out=512,t=t, dropout=dropout, normalization=normalization)
        
        self.Up4 = up_conv(ch_in=512,ch_out=256, dropout=dropout, normalization=normalization)
        self.Att4 = Attention_block(F_g=256,F_l=256,F_int=128, normalization=normalization)
        self.Up_Res4 = Residual_block(ch_in=512, ch_out=256,t=t, dropout=dropout, normalization=normalization)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128, dropout=dropout, normalization=normalization)
        self.Att3 = Attention_block(F_g=128,F_l=128,F_int=64, normalization=normalization)
        self.Up_Res3 = Residual_block(ch_in=256, ch_out=128,t=t, dropout=dropout, normalization=normalization)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64, dropout=dropout, normalization=normalization)
        self.Att2 = Attention_block(F_g=64,F_l=64,F_int=32, normalization=normalization)
        self.Up_Res2 = Residual_block(ch_in=128, ch_out=64,t=t, dropout=dropout, normalization=normalization)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x: torch.Tensor) -> torch.Tensor:
        # encoding path
        x1 = self.Res1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Res2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Res3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Res4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Res5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.Up_Res5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_Res4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_Res3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_Res2(d2)

        d1 = self.Conv_1x1(d2)

        return d1
