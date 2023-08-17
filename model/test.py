import torch
import unittest
from unet import UNet

class TestUNet(unittest.TestCase):
    def setUp(self):
        self.model = UNet(in_channels=3, out_channels=1)
        self.dummy_input = torch.randn(1, 3, 256, 256)
        
    def test_unet_output_shape(self):
        output = self.model(self.dummy_input)
        self.assertEqual(output.shape, (1, 1, 256, 256), "Output shape is incorrect")
    
    def test_unet_forward_pass(self):
        output = self.model(self.dummy_input)
        self.assertIsNotNone(output, "Forward pass result is None")
    
    def test_unet_gradients(self):
        output = self.model(self.dummy_input)
        loss = output.mean()
        loss.backward()
        for param in self.model.parameters():
            self.assertIsNotNone(param.grad, "Gradients are not being computed")
    
    def test_unet_parameter_updates(self):
        output = self.model(self.dummy_input)
        loss = output.mean()
        loss.backward()
        
        initial_params = [param.clone() for param in self.model.parameters()]
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        optimizer.step()
        
        for old_param, new_param in zip(initial_params, self.model.parameters()):
            self.assertFalse(torch.all(old_param == new_param), "Model parameters are not being updated")
    
    def test_unet_parameter_shapes(self):
        for name, param in self.model.named_parameters():
            self.assertNotEqual(param.shape, (), f"Parameter {name} has an empty shape")
    
    def test_unet_segmentation_consistency(self):
        self.model.eval()
        output = self.model(self.dummy_input)
        
        self.assertTrue(torch.all(output >= 0), "Output values are less than 0")
        self.assertTrue(torch.all(output <= 1), "Output values are greater than 1")
        
        binary_mask = (output > 0.5).float()
        self.assertTrue(torch.all(torch.logical_or(binary_mask == 0, binary_mask == 1)), "Binary mask values are not binary")
    
    def test_unet_segmentation_visualization(self):
        self.model.eval()
        output = self.model(self.dummy_input)
        
        segmentation = output.detach().cpu().numpy()
        self.assertEqual(segmentation.shape, (1, 1, 256, 256), "Segmentation shape is incorrect")
        
        import matplotlib.pyplot as plt
        plt.imshow(segmentation[0, 0], cmap='gray')
        plt.title("Segmentation")
        plt.show()

if __name__ == "__main__":
    unittest.main()
