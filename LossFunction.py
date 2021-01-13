import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, target, net_output, smooth=1e-5):
        batch_size = target.size(0)
        # Get probability
        net_output = torch.sigmoid(net_output)
        # Flatten
        target_ = target.view(batch_size, -1)
        net_output_ = net_output.view(batch_size, -1)

        # Calculate Dice
        intersection = (target_ * net_output_).sum(1)
        dice = (2. * intersection + smooth) / (target_.sum(1) + net_output_.sum(1) + smooth)
        return 1 - dice.mean()


class BCE_DiceLoss(nn.Module):
    def __init__(self):
        super(BCE_DiceLoss, self).__init__()

    def forward(self, target, net_output, alpha=0.5, beta=1.0, smooth=1e-5):
        batch_size = target.size(0)
        # Get probability
        net_output = torch.sigmoid(net_output)
        # Flatten
        target_ = target.view(batch_size, -1)
        net_output_ = net_output.view(batch_size, -1)
        # Calculate BCE_DiceLoss
        bce = F.binary_cross_entropy(target_, net_output_)
        intersection = (target_ * net_output_).sum(1)
        dice = (2. * intersection + smooth) / (target_.sum(1) + net_output_.sum(1) + smooth)

        return beta * (1 - dice.mean()) + alpha * bce








