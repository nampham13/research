import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskLoss(nn.Module):
    def __init__(self, depth_weight=1.0, segmentation_weight=1.0, keypoint_weight=1.0, vector_field_weight=1.0, distillation_weight=1.0):
        super(MultiTaskLoss, self).__init__()
        self.depth_loss = nn.MSELoss()
        self.segmentation_loss = nn.CrossEntropyLoss()
        self.keypoint_loss = nn.MSELoss()
        self.vector_field_loss = nn.SmoothL1Loss()
        self.depth_weight = depth_weight
        self.segmentation_weight = segmentation_weight
        self.keypoint_weight = keypoint_weight
        self.vector_field_weight = vector_field_weight
        self.distillation_weight = distillation_weight
        self.distillation_loss = nn.MSELoss()  # Added for pose distillation

    def forward(self, predictions, targets):
        depth_pred = predictions["depth"]
        segmentation_pred = predictions["segmentation"]
        keypoints_pred = predictions["keypoints"]
        vector_field_pred = predictions["vector_field"]
        # Added distillation (pose) prediction
        pose_pred = predictions.get("pose", None)
        
        depth_target = targets["depth"]
        segmentation_target = targets["segmentation"].long()
        keypoints_target = targets["keypoints"]
        vector_field_target = targets.get("vector_field", torch.zeros_like(vector_field_pred))
        # Added pose target extraction if provided
        pose_target = targets.get("pose", None)
        
        # Resize predictions if needed.
        depth_pred = F.interpolate(depth_pred, size=depth_target.shape[2:], mode='bilinear', align_corners=False)
        if segmentation_pred.shape[2:] != segmentation_target.shape[1:]:
            segmentation_pred = F.interpolate(segmentation_pred, size=segmentation_target.shape[1:], mode='bilinear', align_corners=False)
        
        # Resize keypoints target to match prediction resolution if needed.
        if keypoints_target.shape[2:] != keypoints_pred.shape[2:]:
            keypoints_target = F.interpolate(keypoints_target, size=keypoints_pred.shape[2:], mode='bilinear', align_corners=False)
        
        k_loss = self.keypoint_loss(keypoints_pred, keypoints_target)
        
        # Compute center (vector field) loss using instance mask if provided.
        if "instance_mask" in targets:
            instance_mask = targets["instance_mask"]  # assume shape (B, 1, H, W) with ones for object pixels.
            eps = 1e-6
            # Multiply predictions and target by the mask.
            vf_loss = F.smooth_l1_loss(vector_field_pred * instance_mask, vector_field_target * instance_mask, reduction="sum") / (instance_mask.sum() + eps)
        else:
            vf_loss = self.vector_field_loss(vector_field_pred, vector_field_target)
        
        d_loss = self.depth_loss(depth_pred, depth_target)
        s_loss = self.segmentation_loss(segmentation_pred, segmentation_target)
        
        # Compute distillation (pose) loss if provided
        if pose_pred is not None and pose_target is not None:
            distill_loss = self.distillation_loss(pose_pred, pose_target)
        else:
            distill_loss = 0.0

        total_loss = (self.depth_weight * d_loss +
                      self.segmentation_weight * s_loss +
                      self.keypoint_weight * k_loss +
                      self.vector_field_weight * vf_loss +
                      self.distillation_weight * distill_loss)
        
        return total_loss, {
            "depth_loss": d_loss,
            "segmentation_loss": s_loss,
            "keypoint_loss": k_loss,
            "vector_field_loss": vf_loss,
            "distillation_loss": distill_loss
        }
