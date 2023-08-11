import torch
from torchviz import make_dot

from unet import UNet

model = UNet()

dummy_input = torch.randn(1, 3, 572, 572)

output = model(dummy_input)
graph = make_dot(output, params=dict(model.named_parameters()))

# Save the graph visualization to a file
graph.render("unet_graph", format="png")
