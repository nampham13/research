import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskBasedConv(nn.Module):
    def __init__(self, in_channels):
        super(MaskBasedConv, self).__init__()
        # A convolution layer to refine the masked features.
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feature_map, mask):
        """
        Apply mask-based convolution to refine task-specific outputs.
        
        Args:
            feature_map (torch.Tensor): Feature map from a decoder of shape (B, C, H, W).
            mask (torch.Tensor): Segmentation mask of shape (B, 1, H, W) or (B, C, H, W).
                                It should have values between 0 and 1.
        
        Returns:
            torch.Tensor: Refined feature map after applying the mask and convolution.
        """
        # If mask has one channel and feature_map has more channels, expand the mask.
        if mask.size(1) == 1 and feature_map.size(1) > 1:
            mask = mask.expand(-1, feature_map.size(1), -1, -1)
        
        # Ensure mask spatial dimensions match the feature_map.
        if mask.shape[-2:] != feature_map.shape[-2:]:
            mask = F.interpolate(mask, size=feature_map.shape[-2:], mode='bilinear', align_corners=False)
        
        # Apply the mask element-wise.
        masked_features = feature_map * mask
        
        # Refine the masked features using a convolution layer.
        refined_features = self.relu(self.conv(masked_features))
        return refined_features