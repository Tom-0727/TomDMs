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

# Standard UNet with Padding
class UNet(nn.Module):
    def __init__(self, 
                 in_channels: int = 1,
                 out_channels: int = 1,
                 Transpose: bool = True):
        super(UNet, self).__init__()
        
        # Modules in Contracting Path
        self.inconv = DoubleConv(in_channels=in_channels, out_channels=64)
        self.downblock1 = DownBlock(64, 128)
        self.downblock2 = DownBlock(128, 256)
        self.downblock3 = DownBlock(256, 512)
        self.drop3 = nn.Dropout2d(p=0.5)
        self.downblock4 = DownBlock(512, 1024)
        self.drop4 = nn.Dropout2d(p=0.5)

        # Modules in Expansive Path
        self.upblock1 = UpBlock(1024, 512, Transpose=Transpose)
        self.upblock2 = UpBlock(512, 256, Transpose=Transpose)
        self.upblock3 = UpBlock(256, 128, Transpose=Transpose)
        self.upblock4 = UpBlock(128, 64, Transpose=Transpose)

        # Modules in Output
        self.outconv = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def forward(self, x):
        # Contracting Path
        x1 = self.inconv(x)
        x2 = self.downblock1(x1)
        x3 = self.downblock2(x2)
        x4 = self.downblock3(x3)
        x4 = self.drop3(x4)
        x5 = self.downblock4(x4)
        x5 = self.drop4(x5)

        # Expansive Path
        x = self.upblock1(x5, x4)
        x = self.upblock2(x, x3)
        x = self.upblock3(x, x2)
        x = self.upblock4(x, x1)
        x = self.outconv(x)

        return x