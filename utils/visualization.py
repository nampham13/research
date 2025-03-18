import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def visualize_rgb(image, title="RGB Image"):
    """
    Visualize the input RGB image.
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()

def visualize_segmentation(image, segmentation_mask, title="Segmentation"):
    """
    Visualize segmentation mask overlaid on the input image.
    """
    overlay = image.copy()
    segmentation_mask_colored = cv2.applyColorMap((segmentation_mask * 255).astype(np.uint8), cv2.COLORMAP_JET)
    blended = cv2.addWeighted(overlay, 0.6, segmentation_mask_colored, 0.4, 0)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()

def visualize_depth(depth_map, title="Depth Map"):
    """
    Visualize the predicted depth map.
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(depth_map, cmap='plasma')
    plt.colorbar(label="Depth")
    plt.title(title)
    plt.axis("off")
    plt.show()

def visualize_keypoints(image, keypoints, title="Keypoints"):
    """
    Visualize keypoints overlaid on the image.
    """
    image_copy = image.copy()
    for x, y in keypoints:
        cv2.circle(image_copy, (int(x), int(y)), 5, (0, 255, 0), -1)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()

def visualize_vector_fields(image, vector_field, step=16, scale=10, title="Vector Field"):
    """
    Visualize vector fields overlaid on the image.
    """
    h, w = vector_field.shape[:2]
    y, x = np.meshgrid(np.arange(0, h, step), np.arange(0, w, step))
    u = vector_field[::step, ::step, 0]  # X-component
    v = vector_field[::step, ::step, 1]  # Y-component

    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.quiver(x, y, u, v, color="blue", angles="xy", scale_units="xy", scale=scale)
    plt.title(title)
    plt.axis("off")
    plt.show()
