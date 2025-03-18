import torch
import torch.nn as nn
import torch.nn.functional as F
from models.shared_encoder import SharedEncoder

##############################################
# Cross Task Attention Module
##############################################
class CrossTaskAttention(nn.Module):
    def __init__(self, in_channels):
        """
        A simple self-attention module using a query-key-value mechanism.
        Args:
            in_channels (int): Number of channels in the feature maps.
        """
        super(CrossTaskAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv   = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x, context):
        """
        Args:
            x: primary feature map [B, C, H, W]
            context: feature map from the other task [B, C, H, W]
        Returns:
            Fused feature map after applying cross-task attention.
        """
        B, C, H, W = x.shape
        # Generate query and key
        proj_query = self.query_conv(x).view(B, -1, H * W)      # [B, C//8, N]
        proj_key   = self.key_conv(context).view(B, -1, H * W)    # [B, C//8, N]
        # Compute attention weights. Transpose query to [B, N, C//8]
        energy = torch.bmm(proj_query.transpose(1, 2), proj_key)  # [B, N, N]
        attention = F.softmax(energy, dim=-1)  # [B, N, N]
        # Generate value transformations.
        proj_value = self.value_conv(context).view(B, -1, H * W)  # [B, C, N]
        out = torch.bmm(proj_value, attention.transpose(1, 2))    # [B, C, N]
        out = out.view(B, C, H, W)
        # Fuse with original feature map.
        out = self.gamma * out + x
        return out

#############################################
# 1. Semantic Segmentation Decoder (U-Net)  #
#############################################
class UNetSegmentationDecoder(nn.Module):
    def __init__(self, in_channels, num_classes):
        """
        A simple U-Net–style decoder for semantic segmentation enhanced with cross-task attention.
        Args:
            in_channels (int): Number of channels in the input feature map.
            num_classes (int): Number of segmentation classes.
        """
        super(UNetSegmentationDecoder, self).__init__()
        # First upsampling block
        self.up1 = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        # Second upsampling block
        self.up2 = nn.ConvTranspose2d(in_channels // 2, in_channels // 4, kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        # Cross-task attention block (optional)
        self.cross_att = CrossTaskAttention(in_channels // 4)
        # Final prediction layer
        self.final_conv = nn.Conv2d(in_channels // 4, num_classes, kernel_size=1)
        
    def forward(self, x, depth_feat=None):
        """
        Args:
            x: features for segmentation [B, in_channels, H, W]
            depth_feat (optional): features from the depth decoder to guide segmentation.
        """
        x = self.up1(x)
        x = self.conv1(x)
        x = self.up2(x)
        x = self.conv2(x)
        if depth_feat is not None:
            # Assume depth_feat is resized to match x dimensions.
            x = self.cross_att(x, depth_feat)
        x = self.final_conv(x)
        return x

#################################################
# 2. Vector-Field Prediction Decoder (PoseCNN)  #
#################################################
class PoseCNNVectorFieldDecoder(nn.Module):
    def __init__(self, in_channels, out_channels=2):
        """
        A decoder inspired by PoseCNN to predict a dense 2D vector field.
        Args:
            in_channels (int): Number of channels in the input feature map.
            out_channels (int): Typically 2 (for x and y vector components).
        """
        super(PoseCNNVectorFieldDecoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, out_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        # Optionally, upsample to a higher resolution if needed.
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.conv4(x)
        x = self.upsample(x)
        return x

#########################################################
# 3. Depth Estimation Decoder (Monocular Depth Methods) #
#########################################################
class MonocularDepthDecoder(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        """
        A decoder for monocular depth estimation enhanced with cross-task attention.
        Args:
            in_channels (int): Number of channels in the input feature map.
            out_channels (int): Number of output channels (usually 1 for depth).
        """
        super(MonocularDepthDecoder, self).__init__()
        # Upsampling block 1
        self.up1 = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        # Upsampling block 2
        self.up2 = nn.ConvTranspose2d(in_channels // 2, in_channels // 4, kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        # Cross-task attention block (optional)
        self.cross_att = CrossTaskAttention(in_channels // 4)
        # Final prediction layer
        self.final_conv = nn.Conv2d(in_channels // 4, out_channels, kernel_size=1)
    
    def forward(self, x, seg_feat=None):
        """
        Args:
            x: features for depth [B, in_channels, H, W]
            seg_feat (optional): features from the segmentation decoder to guide depth estimation.
        """
        x = self.up1(x)
        x = self.conv1(x)
        x = self.up2(x)
        x = self.conv2(x)
        if seg_feat is not None:
            # Assume seg_feat is resized to match x dimensions.
            x = self.cross_att(x, seg_feat)
        x = self.final_conv(x)
        return x
