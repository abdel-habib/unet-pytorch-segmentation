from unet import UNet
import torch

def test():
    # batch_size=3, in_channels=1, 160x160
    x = torch.randn((3, 1, 160, 160))

    model = UNet(in_channels=1, out_channels=1)
    preds = model(x)


    print(f"input shape: {x.shape}, prediction shape: {preds.shape}")
    assert preds.shape == x.shape, "Shapes are not identical for input and prediction"

if __name__ == "__main__":
    test()