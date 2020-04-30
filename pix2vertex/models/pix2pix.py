import torch
import torch.nn as nn

def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.LeakyReLU(0.2,inplace=True),
        nn.Conv2d(in_channels, out_channels, 4, stride=2,padding=1),
        nn.BatchNorm2d(out_channels)
    )


def deconv_block(in_channels, out_channels,use_dropout=False):
    layers = [
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2,padding=1),
        nn.BatchNorm2d(out_channels)
    ]
    if use_dropout:
        layers.append(nn.Dropout(0.5))
    return nn.Sequential(*layers)

class UNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv_down1 = nn.Conv2d(3,64,4,stride=2,padding=1)

        self.conv_down2 = conv_block(64,128)

        self.conv_down3 = conv_block(128,256)

        self.conv_down4 = conv_block(256,512)

        self.conv_down5 = conv_block(512,512)

        self.conv_down6 = conv_block(512,512)

        self.conv_down7 = conv_block(512,512)

        self.conv_down8 = conv_block(512,512)

        self.conv_up1 = deconv_block(512,512,use_dropout=True)

        self.conv_up2 = deconv_block(1024,512,use_dropout=True)

        self.conv_up3 = deconv_block(1024,512,use_dropout=True)

        self.conv_up4 = deconv_block(1024,512)

        self.conv_up5 = deconv_block(1024,256)

        self.conv_up6 = deconv_block(512,128)

        self.conv_up7 = deconv_block(256,64)

        self.conv_up8 = deconv_block(128,64)

        self.conv_up9 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, 3, stride=1,padding=1),
            nn.BatchNorm2d(64)
        )

        self.conv_up9 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, 3, stride=1,padding=1),
            nn.BatchNorm2d(64)
        )

        self.conv_up10 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 3, stride=1,padding=1),
            nn.BatchNorm2d(32)
        )

        self.conv_up11 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 7, 3, stride=1,padding=1)
        )

    # TODO: rewrite nicely

    def forward(self, x):
        down1 = self.conv_down1(x)
        down2 = self.conv_down2(down1)
        down3 = self.conv_down3(down2)
        down4 = self.conv_down4(down3)
        down5 = self.conv_down5(down4)
        down6 = self.conv_down6(down5)
        down7 = self.conv_down7(down6)
        down8 = self.conv_down8(down7)

        up1 = self.conv_up1(down8)
        up2 = self.conv_up2(torch.cat([up1, down7], dim=1))
        up3 = self.conv_up3(torch.cat([up2, down6], dim=1))
        up4 = self.conv_up4(torch.cat([up3, down5], dim=1))
        up5 = self.conv_up5(torch.cat([up4, down4], dim=1))
        up6 = self.conv_up6(torch.cat([up5, down3], dim=1))
        up7 = self.conv_up7(torch.cat([up6, down2], dim=1))
        up8 = self.conv_up8(torch.cat([up7, down1], dim=1))
        up9 = self.conv_up9(up8)
        up10 = self.conv_up10(up9)
        up11 = self.conv_up11(up10)

        return up11
