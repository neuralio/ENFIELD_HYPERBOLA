import torch
import random
import numpy as np
import albumentations as A
import cv2

def apply_to_each_channel(x, transform):
    """
    Applies a transformation to each channel of a multi-channel tensor.
    """
    channels = [transform(x[i:i+1]) for i in range(x.shape[0])]
    return torch.cat(channels, dim=0)

# Define the spatial transforms (applied to both image and mask)
spatial_transform = A.Compose([
    A.Affine(
        translate_percent={'x': (0.02, 0.08), 'y': (0.02, 0.08)},
        scale=(0.9, 1.2),
        rotate=(-15, 15),
        #shear=(-5, 5),
        border_mode=cv2.BORDER_CONSTANT,
        p=1.0
    ),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),
], p=1.0)

# Define the image-only transforms (applied only to the image)
image_only_transform = A.Compose([
    A.GaussianBlur(blur_limit=(3, 5), sigma_limit=(0.1, 1.0), p=0.5),
    # A.CoarseDropout(num_holes_range=(3, 6),
    #                 hole_height_range=(10, 20), 
    #                 hole_width_range=(10, 20), 
    #                 fill="random_uniform", p=1.0),
    A.ColorJitter(
        brightness=(0.8, 1.2),
        contrast=(0.8, 1.2),
        saturation=(0.8, 1.2),
        hue=(-0.05, 0.05),
        p=1.0
    ),
], p=1.0)

# Updated DualTransform using albumentations
class DualTransform:
    def __init__(self, spatial_transform, image_only_transform=None):
        self.spatial_transform = spatial_transform
        self.image_only_transform = image_only_transform

    def __call__(self, img, mask):
        # Generate a random seed to ensure the same spatial transform for image and mask
        seed = random.randint(0, 2**32 - 1)

        # Convert PyTorch tensors to NumPy for albumentations
        img_np = img.permute(1, 2, 0).numpy()  # [C, H, W] -> [H, W, C]
        mask_np = mask.numpy()  # [H, W]

        # Apply spatial transforms to both image and mask with the same seed
        random.seed(seed)
        augmented = self.spatial_transform(image=img_np, mask=mask_np)
        img_np = augmented['image']  # Shape: [H, W, C]
        mask_np = augmented['mask']  # Shape: [H, W]

        # Apply image-only transforms to the image (not the mask)
        if self.image_only_transform:
            random.seed()  # Reset the seed for image-only transforms

            # Handle ColorJitter for 10-channel hyperspectral image
            # Select first 3 channels (or average) for ColorJitter, then stack back
            if img_np.shape[-1] > 3:  # More than 3 channels (e.g., 10 channels)
                # Option 1: Select first 3 channels
                img_3ch = img_np[:, :, :3]  # [H, W, 3]
                remaining_channels = img_np[:, :, 3:]  # [H, W, 7]

                # Apply image-only transforms to the 3-channel image
                img_only = self.image_only_transform(image=img_3ch, mask=None)
                img_3ch_transformed = img_only['image']  # [H, W, 3]

                # Stack the transformed 3 channels with the remaining channels
                img_np = np.concatenate([img_3ch_transformed, remaining_channels], axis=-1)  # [H, W, 10]
            else:
                # If image is already 1 or 3 channels, apply directly
                img_only = self.image_only_transform(image=img_np, mask=None)
                img_np = img_only['image']

        # Convert back to PyTorch tensors
        img = torch.from_numpy(img_np).permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
        mask = torch.from_numpy(mask_np).long()  # [H, W]

        return img, mask
    
# Instantiate the transform
transformations = DualTransform(spatial_transform, image_only_transform)