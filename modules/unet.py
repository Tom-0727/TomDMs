from .units import *


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