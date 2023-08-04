# Main difference is that this implementation is done using padded conv
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.conv(x)
    
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512], ):
        super(UNet, self).__init__()
        '''
        in_channels is the number of input channels of the segmentation, color RGB image = 3 channels
        out_channels is the number of output channels of the segmentation, binary = 1
        '''

        self.ups    = nn.ModuleList()
        self.downs  = nn.ModuleList()
        self.pool   = nn.MaxPool2d(kernel_size=2, stride=2)

        # downsampling
        for feature in features:
            # map the input channel to the next feature channel
            self.downs.append(DoubleConv(in_channels=in_channels, out_channels=feature))

            # reset the input channel to the feature channel
            in_channels=feature
            
        # upsampling
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    in_channels=feature*2, out_channels=feature, kernel_size=2, stride=2
                )
            )
            self.ups.append(
                DoubleConv(
                in_channels=feature*2, out_channels=feature
                )
            )

        # bottleneck layer
        self.bottleneck = DoubleConv(in_channels=features[-1], out_channels=features[-1]*2)

        # final output layer
        self.final_conv = nn.Conv2d(in_channels=features[0], out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            # step of 2 is to make up then DoubleConv 
            # upsampling
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                # can crop, pad, or resize to handle different concatination sizes
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            
            # DoubleConv
            self.ups[idx+1](concat_skip)

        return self.final_conv(x)
    
def test():
    # batch_size=3, n_channels=1, 160x160
    x = torch.randn((3, 1, 160, 160))

    model = UNet(in_channels=1, out_channels=1)
    preds = model(x)

    print(preds.shape, x.shape)
    assert preds.shape == x.shape

if __name__ == "__main__":
    test()