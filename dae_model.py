import numpy as np
import torch
import torch.cuda as cuda
import torch.nn as nn

# # DAE

# Encoder 
# torch.nn.Conv2d(in_channels, out_channels, kernel_size,
#                 stride=1, padding=0, dilation=1,
#                 groups=1, bias=True)
# batch x 2 x 320 x 320 -> batch x 204800

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(2,32,3,padding=1),   # batch x 32 x 320 x 320
            nn.LeakyReLU(negative_slope=args.reluslope),
            nn.BatchNorm2d(32),
            nn.Conv2d(32,32,3,padding=1),   # batch x 32 x 320 x 320
            nn.LeakyReLU(negative_slope=args.reluslope),
            nn.BatchNorm2d(32),
            nn.Conv2d(32,64,3,padding=1),  # batch x 64 x 320 x 320
            nn.LeakyReLU(negative_slope=args.reluslope),
            nn.BatchNorm2d(64),
            nn.Conv2d(64,64,3,padding=1),  # batch x 64 x 320 x 320
            nn.LeakyReLU(negative_slope=args.reluslope),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2,2)   # batch x 64 x 160 x 160
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64,128,3,padding=1),  # batch x 128 x 160 x 160
            nn.LeakyReLU(negative_slope=args.reluslope),
            nn.BatchNorm2d(128),
            nn.Conv2d(128,128,3,padding=1),  # batch x 128 x 160 x 160
            nn.LeakyReLU(negative_slope=args.reluslope),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2,2),               # batch x 128 x 80 x 80
            nn.Conv2d(128,256,3,padding=1),  # batch x 256 x 80 x 80
            nn.LeakyReLU(negative_slope=args.reluslope),
        )
        
                
    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(x.shape[0], -1)
        return out

# Decoder 
# torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
#                          stride=1, padding=0, output_padding=0,
#                          groups=1, bias=True)
# output_height = (height-1)*stride + kernel_size - 2*padding + output_padding
# batch x 256 x 80 x 80 -> batch x 1638400

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(256,128,3,2,1,1),     # batch x 128 x 160 x 160
            nn.LeakyReLU(negative_slope=args.reluslope),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128,128,3,1,1),      # batch x 128 x 160 x 160
            nn.LeakyReLU(negative_slope=args.reluslope),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128,64,3,1,1),       # batch x 64 x 160 x 160
            nn.LeakyReLU(negative_slope=args.reluslope),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64,64,3,1,1),        # batch x 64 x 160 x 160
            nn.LeakyReLU(negative_slope=args.reluslope),
            nn.BatchNorm2d(64)
        )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(64,32,3,1,1),        # batch x 32 x 160 x 160
            nn.LeakyReLU(negative_slope=args.reluslope),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32,32,3,1,1),        # batch x 32 x 160 x 160
            nn.LeakyReLU(negative_slope=args.reluslope),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32,2,3,2,1,1),        # batch x 2 x 320 x 320
            nn.LeakyReLU(negative_slope=args.reluslope),
        )
        
    def forward(self,x):
        out = x.view(x.shape[0],256,80,80)
        out = self.layer1(out)
        out = self.layer2(out)
        return out

