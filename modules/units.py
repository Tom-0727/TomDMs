import torch
import torch.nn as nn


# Double Convolution With Padding
class DoubleConv(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int,
                 mid_channels: int = None) -> None:
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


# Down Sampling Block
class DownBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int):
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


# Up Sampling Block
class UpBlock(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int,
                 Transpose: bool = False):
        super(UpBlock, self).__init__()
        if Transpose:
            self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        else:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, in_channels//2, kernel_size=1, padding=0),  # 1x1 Convolution to halve channels
                nn.ReLU(inplace=True)
            )
        self.dconv = DoubleConv(in_channels, out_channels)
        self.up.apply(self.init_weights)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = nn.functional.pad(x1, [diffX//2, diffX-diffX//2, diffY//2, diffY-diffY//2])  # pad zeros to left,right,top & down side
        x = torch.cat([x2, x1], dim=1)
        return self.dconv(x)

    @staticmethod
    def init_weights(m):
        if type(m) == nn.Conv2d:
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
    

