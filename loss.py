import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        """
        Initialize the DiceLoss module.
        
        Args:
            smooth (float): A smoothing term added to the numerator and denominator
                           to prevent division by zero.
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        """
        Compute the Dice loss between predicted inputs and target masks.
        
        Args:
            inputs (torch.Tensor): Predicted values (usually after a sigmoid activation).
            targets (torch.Tensor): Target binary masks.
            
        Returns:
            torch.Tensor: Dice loss.
        """
        inputs = torch.sigmoid(inputs)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice

class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1):
        """
        Initialize the DiceBCELoss module.
        
        Args:
            smooth (float): A smoothing term added to the numerator and denominator
                           to prevent division by zero.
        """
        super(DiceBCELoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        """
        Compute the combined Dice loss and Binary Cross-Entropy (BCE) loss.
        
        Args:
            inputs (torch.Tensor): Predicted values (usually after a sigmoid activation).
            targets (torch.Tensor): Target binary masks.
            
        Returns:
            torch.Tensor: Combined loss.
        """
        inputs = torch.sigmoid(inputs)
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='mean')
        dice_bce_loss = bce_loss + dice_loss
        return dice_bce_loss
