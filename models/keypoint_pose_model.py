import torch
import torch.nn as nn
import torch.nn.functional as F

class KeypointPoseDecoder(nn.Module):
    def __init__(self, in_channels, num_keypoints):
        """
        Keypoint pose decoder for detecting keypoints and estimating object pose.
        
        Args:
            in_channels (int): Number of input channels from the encoder.
            num_keypoints (int): Number of keypoints to predict.
        """
        super(KeypointPoseDecoder, self).__init__()
        
        # Keypoint heatmap prediction layers
        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, num_keypoints, kernel_size=1)  # Output: heatmaps

        # Pose regression layers (optional, for 6D pose estimation)
        self.fc1 = nn.Linear(in_channels, 256)
        self.fc2 = nn.Linear(256, 6)  # Output: 3D rotation (R) and translation (T)

    def forward(self, features):
        """
        Forward pass of the keypoint pose decoder.
        
        Args:
            features (torch.Tensor): Input feature map from the shared encoder.
            
        Returns:
            keypoint_heatmaps (torch.Tensor): Heatmaps for keypoint locations.
            pose_params (torch.Tensor): Rotation and translation parameters (optional).
        """
        # Heatmap prediction
        x = F.relu(self.conv1(features))
        x = F.relu(self.conv2(x))
        keypoint_heatmaps = self.conv3(x)  # Shape: (B, num_keypoints, H, W)

        # Flatten feature map for pose estimation
        flattened_features = torch.mean(features, dim=(2, 3))  # Global average pooling
        pose_params = F.relu(self.fc1(flattened_features))
        pose_params = self.fc2(pose_params)  # Shape: (B, 6)

        return keypoint_heatmaps, pose_params


