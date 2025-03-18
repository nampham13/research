import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from models.decoder import UNetSegmentationDecoder, PoseCNNVectorFieldDecoder
from models.mask_based_convolution import MaskBasedConv

class MultiTaskModel(nn.Module):
    """
    Multi-task model for depth estimation, segmentation, keypoint prediction and 6D pose estimation.
    """
    def __init__(self, num_classes=21):
        super(MultiTaskModel, self).__init__()
        # Shared backbone: ResNet50 (excluding final fc layer)
        resnet = resnet50(pretrained=True)
        self.shared_encoder = nn.Sequential(*list(resnet.children())[:-2])
        
        # Task decoders (assumes shared features of 2048 channels)
        self.segmentation_decoder = UNetSegmentationDecoder(in_channels=2048, num_classes=num_classes)
        self.vector_field_decoder = PoseCNNVectorFieldDecoder(in_channels=2048)
        
        # Refinement layers via mask-based convolution.
        self.mask_based_conv_depth = MaskBasedConv(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        self.mask_based_conv_keypoints = MaskBasedConv(in_channels=2, out_channels=2, kernel_size=3, padding=1)
        
        # Multi-modal distillation: fuse intermediate features.
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(4, 512, kernel_size=1),  # changed from 2048*3 to 4 channels
            nn.ReLU(inplace=True)
        )
        # 6D Pose estimation head from fused features.
        self.pose_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 6)  # 3D rotation and translation.
        )
    
    def forward(self, x):
        features = self.shared_encoder(x)
        
        segmentation_logits = self.segmentation_decoder(features)
        seg_mask = segmentation_logits.softmax(dim=1).mean(dim=1, keepdim=True)
        
        depth = self.depth_decoder(features, seg_mask)
        vector_field = self.vector_field_decoder(features, seg_mask)
        
        # Resize segmentation mask to match depth and vector_field resolutions.
        depth_mask = F.interpolate(seg_mask, size=(depth.shape[2], depth.shape[3]), mode='bilinear', align_corners=False)
        vf_mask = F.interpolate(seg_mask, size=(vector_field.shape[2], vector_field.shape[3]), mode='bilinear', align_corners=False)
        
        refined_depth = self.mask_based_conv_depth(depth, depth_mask)
        refined_keypoints = self.mask_based_conv_keypoints(vector_field, vf_mask)
        
        # Fuse features from decoders.
        fusion = torch.cat([refined_depth, refined_keypoints, depth_mask], dim=1)
        fused_features = self.fusion_conv(fusion)
        pose_params = self.pose_head(fused_features)
        
        return {
            "depth": refined_depth,
            "segmentation": segmentation_logits,
            "keypoints": refined_keypoints,
            "vector_field": vector_field,
            "pose": pose_params
        }
