import numpy as np

def compute_adds(pred_pose, gt_pose, model_points):
    def transform(pose, points):
        R = pose[:, :3]
        t = pose[:, 3]
        return (R @ points.T).T + t
    pred_trans = transform(pred_pose, model_points)
    gt_trans = transform(gt_pose, model_points)
    return np.mean(np.linalg.norm(pred_trans - gt_trans, axis=1))

def compute_reprojection_error(pred_pose, gt_pose, cam_K, model_points):
    def project(pose, points):
        R = pose[:, :3]
        t = pose[:, 3]
        points_trans = (R @ points.T).T + t
        points_proj = (cam_K @ points_trans.T).T
        points_proj = points_proj / points_proj[:, 2:3]
        return points_proj[:, :2]
    pred_proj = project(pred_pose, model_points)
    gt_proj = project(gt_pose, model_points)
    return np.mean(np.linalg.norm(pred_proj - gt_proj, axis=1))



