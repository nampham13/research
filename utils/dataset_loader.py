import os
from PIL import Image
from torch.utils.data import Dataset

class MultiTaskDataset(Dataset):
    def __init__(self, rgb_dir, depth_dir, mask_dir, transform=None, depth_transform=None, mask_transform=None):
        self.rgb_dir = rgb_dir
        self.depth_dir = depth_dir
        self.mask_dir = mask_dir
        self.rgb_paths = sorted([os.path.join(rgb_dir, f) for f in os.listdir(rgb_dir)
                                  if f.lower().endswith(('.png','.jpg','.jpeg'))])
        self.depth_paths = sorted([os.path.join(depth_dir, f) for f in os.listdir(depth_dir)
                                    if f.lower().endswith(('.png','.jpg','.jpeg'))])
        self.mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir)
                                   if f.lower().endswith(('.png','.jpg','.jpeg'))])
        self.transform = transform
        self.depth_transform = depth_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return min(len(self.rgb_paths), len(self.depth_paths), len(self.mask_paths))
    
    def __getitem__(self, idx):
        rgb_img = Image.open(self.rgb_paths[idx]).convert("RGB")
        depth_img = Image.open(self.depth_paths[idx]).convert("L")
        mask_img = Image.open(self.mask_paths[idx]).convert("L")
        if self.transform:
            rgb_img = self.transform(rgb_img)
        if self.depth_transform:
            depth_img = self.depth_transform(depth_img)
        if self.mask_transform:
            mask_img = self.mask_transform(mask_img)
        return rgb_img, depth_img, mask_img
