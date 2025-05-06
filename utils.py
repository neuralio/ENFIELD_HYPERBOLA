import os
import numpy as np
import json
import torch
import tabulate
import random
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, \
    confusion_matrix, matthews_corrcoef, accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, default_collate
from torchvision.transforms.v2 import CutMix, MixUp, RandomChoice
from torch_geometric.data import HeteroData, Batch
from matplotlib_config import configure_matplotlib
from metrics import dice_iou_score_multiclass_weighted
from config import NB_CLASSES, OUTPUT_DIR
# Apply global settings
configure_matplotlib()

cutmix = CutMix(num_classes=NB_CLASSES, alpha=0.2)
mixup = MixUp(num_classes=NB_CLASSES, alpha=0.2)
cutmix_or_mixup = RandomChoice([cutmix, mixup])

def collate_fn_mixup(batch):
    mixup_prob=0.5
    # Stack images and labels normally.
    collated = default_collate(batch)
    images, masks, img_idx = collated  # images: (B, C, H, W), masks: (B, H, W)
    
    # Apply the built-in transform only on images.
    # First, decide whether to apply mixup.
    if np.random.rand() < mixup_prob:
        # The transforms expect labels in classification format,
        # so if we only mix images, we can simply call the transform on images.
        # One workaround is to pass dummy labels that won't be used.
        dummy_labels = torch.zeros(images.size(0), NB_CLASSES)  # dummy classification labels
        mixed_images, _ = cutmix_or_mixup(images, dummy_labels)
        images = mixed_images

    return images, masks, img_idx

def collate_fn(batch):
    imgs, labels, img_idx = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    labels = torch.stack(labels, dim=0)
    img_idx = torch.tensor(img_idx)  # Convert to tensor for batch processing
    return imgs, labels, img_idx

def collate_fn_graph(batch):
    imgs, labels, img_idx, hetero_graph = zip(*batch) 
    imgs = torch.stack(imgs, dim=0)
    labels = torch.stack(labels, dim=0)
    img_idxs = torch.tensor(img_idx)
    batched_hetero_graph = Batch.from_data_list(hetero_graph)
    return imgs, labels, img_idxs, batched_hetero_graph

    
class PatchDataDataset(Dataset):
    def __init__(self,  
                 data_files,
                 label_files, 
                 patch_size, 
                 shuffle,
                 num_classes,
                 overlap_step,
                 target_height,
                 target_width,
                 selected_channels,
                 model_name,
                 transform=None):
        self.data_files = data_files
        self.label_files = label_files
        self.patch_size = patch_size
        self.shuffle = shuffle
        self.num_classes = num_classes
        self.overlap_step = overlap_step
        self.target_height = target_height
        self.target_width = target_width
        self.transform = transform
        self.model_name = model_name
        self.selected_channels = selected_channels
        self.original_data_files = data_files.copy()  # Store original file paths
        self.original_label_files = label_files.copy()
        self.patch_mappings = []  # List of (img_idx, start_x, start_y)
    
        self.saved_count = 0  #
        self.resized_data = []
        self.resized_labels = []

        # Generate patches and mappings
        for img_idx, (img_file, label_file) in enumerate(zip(self.data_files, self.label_files)):
            # Load image and label
            img_arr = np.load(img_file, mmap_mode='r')
            label_arr = np.load(label_file, mmap_mode='r')
            H, W, C = img_arr.shape

            if H < self.patch_size or W < self.patch_size:
                raise ValueError(f"Image {img_file} dimensions ({H}, {W}) are smaller than patch_size {self.patch_size}")

            # Resize to target dimensions
            img_arr_resized = cv2.resize(img_arr, (self.target_width, self.target_height), interpolation=cv2.INTER_LINEAR)
            label_arr_resized = cv2.resize(label_arr, (self.target_width, self.target_height), interpolation=cv2.INTER_NEAREST)

            img_arr_resized = img_arr_resized[:, :, self.selected_channels]

            # Store resized images separately (do NOT overwrite original paths)
            self.resized_data.append(img_arr_resized)
            self.resized_labels.append(label_arr_resized)

            # Create patch mappings based on overlap step
            num_patches_h = (self.target_height - self.patch_size) // self.overlap_step + 1
            num_patches_w = (self.target_width - self.patch_size) // self.overlap_step + 1

            for i in range(num_patches_h):
                for j in range(num_patches_w):
                    start_x = i * self.overlap_step
                    start_y = j * self.overlap_step
                    self.patch_mappings.append((img_idx, start_x, start_y))
          
        if not self.patch_mappings:
            raise ValueError("No valid patches generated. Check image dimensions and patch_size.")

        # Shuffle the patch list
        self.indices = list(range(len(self.patch_mappings)))
        if self.shuffle:
            random.shuffle(self.indices)

    def __len__(self):
        return len(self.patch_mappings)

    def save_patch(self, img_tensor, label_tensor, index, img_idx):
        """
        Save a visualization of the normalized patch and its label
        """
        if self.saved_count >= 20:
            return
        
        # Convert to numpy for visualization
        img_np = img_tensor.numpy()
        
        # Transpose from (C, H, W) to (H, W, C) for visualization
        img_np = img_np.transpose(1, 2, 0)  # Shape: (48, 48, 5)
        
        # For visualization, we'll just use the values as they are (already 0-1)
        # If using specific bands for visualization, select them here
        
        # For example, visualize using the first channel
        img_band_0 = img_np[:, :, 0]
        
        label_np = label_tensor.numpy()

        img_name = os.path.basename(self.original_data_files[img_idx])
        label_name = os.path.basename(self.original_label_files[img_idx])

        plt.imsave(os.path.join(OUTPUT_DIR, f'augmented_pixel_norm_patch_{index}_img_{img_name}.png'), img_band_0, cmap='gray')
        plt.imsave(os.path.join(OUTPUT_DIR, f'augmented_pixel_norm_patch_{index}_label_{label_name}.png'), label_np, cmap='jet')

        print(f"Saved pixel-normalized patch {index} from image: {self.original_data_files[img_idx]}")

        self.saved_count += 1

    def extract_patch(self, arr, start_x, start_y, patch_size, is_label):
        """
        Extract patch and apply padding if necessary.
        """
        H, W = arr.shape[:2]
        end_x = start_x + patch_size
        end_y = start_y + patch_size
        
        if end_x <= H and end_y <= W:
            # No padding needed — fully within boundaries
            patch = arr[start_x:end_x, start_y:end_y].copy() if is_label else arr[start_x:end_x, start_y:end_y, ...].copy()
        else:
            # Extract the region within bounds
            region = arr[start_x:min(end_x, H), start_y:min(end_y, W)].copy()
            pad_bottom = max(0, end_x - H)
            pad_right = max(0, end_y - W)

            if region.size == 0:
                # If region is empty → Fallback to constant padding
                if is_label:
                    patch = np.pad(
                        region,
                        ((0, pad_bottom), (0, pad_right)),
                        mode='constant',
                        constant_values=0
                    )
                else:
                    patch = np.pad(
                        region,
                        ((0, pad_bottom), (0, pad_right), (0, 0)),
                        mode='constant',
                        constant_values=0
                    )
            else:
                # Region is non-empty → Use 'reflect' for image, 'constant' for label
                if is_label:
                    patch = np.pad(
                        region,
                        ((0, pad_bottom), (0, pad_right)),
                        mode='constant',
                        constant_values=0
                    )
                else:
                    patch = np.pad(
                        region,
                        ((0, pad_bottom), (0, pad_right), (0, 0)),
                        mode='reflect'
                    )
        
        return patch

    def __getitem__(self, index):
        mapped_index = self.indices[index]
        img_idx, start_x, start_y = self.patch_mappings[mapped_index]

        # Extract patches directly
        img_patch = self.extract_patch(self.resized_data[img_idx], start_x, start_y, self.patch_size, is_label=False)
        label_patch = self.extract_patch(self.resized_labels[img_idx], start_x, start_y, self.patch_size, is_label=True)

        img_tensor = torch.from_numpy(img_patch).float().permute(2, 0, 1)
        label_tensor = torch.from_numpy(label_patch).long()

        original_tensor = img_tensor.clone()
        # Apply pixel-wise spectral normalization instead of channel-wise normalization
        img_tensor = normalize_spectral_signature(img_tensor)
        # if index == 0:  # Only for the first patch
        #     visualize_spectral_signatures(original_tensor, img_tensor)

        if self.transform:
            img_tensor, label_tensor = self.transform(img_tensor, label_tensor)
            # if index < 20:
            #     self.save_patch(img_tensor, label_tensor, index, img_idx)

        if self.model_name == 'HybridModel_4':
            hetero_graph = create_hetero_patch_graph(img_tensor)
            return img_tensor, label_tensor, img_idx, hetero_graph
        else:
            return img_tensor, label_tensor, img_idx

    
# SplitPatchDataDataset 
class SplitPatchDataDataset:
    def __init__(self,
                 base_path, 
                 patch_size, 
                 validation_split,
                 num_classes,
                 overlap_step,
                 target_height,
                 target_width,
                 selected_channels,
                 shuffle=True,
                 random_state=42,
                 transform=None):
        self.base_path = base_path
        self.patch_size = patch_size
        self.validation_split = validation_split
        self.num_classes = num_classes
        self.overlap_step = overlap_step
        self.target_height = target_height
        self.target_width = target_width
        self.shuffle = shuffle
        self.random_state = random_state
        self.transform = transform
        self.selected_channels = selected_channels
        self.model_name = None
        # Get file paths
        self.data_files, self.label_files = self.get_file_paths()
        
        # Split train/val
        self.train_data_files, self.val_data_files, self.train_label_files, self.val_label_files = train_test_split(
            self.data_files, self.label_files, test_size=validation_split, random_state=random_state)


    def get_file_paths(self):
        data_files = []
        label_files = []
        for folder in sorted(os.listdir(self.base_path)):
            folder_path = os.path.join(self.base_path, folder)
            data_path = os.path.join(folder_path, "DATA")
            label_path = os.path.join(folder_path, "GROUND-TRUTH LABELS")
            if not os.path.isdir(data_path) or not os.path.isdir(label_path):
                continue
            d_files = sorted([os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith(".npy")])
            l_files = sorted([os.path.join(label_path, f) for f in os.listdir(label_path) if f.endswith(".npy")])
            data_files.extend(d_files)
            label_files.extend(l_files)
        return data_files, label_files

    def get_train_dataset(self):
        if self.model_name is None:
            raise ValueError("Model name must be set using `set_model_name()` before accessing the train dataset.")
        
        return PatchDataDataset(self.train_data_files, 
                                self.train_label_files, 
                                patch_size=self.patch_size, 
                                shuffle=self.shuffle, 
                                num_classes=self.num_classes,
                                overlap_step=self.overlap_step,
                                target_height=self.target_height,
                                target_width=self.target_width,
                                selected_channels=self.selected_channels,
                                model_name=self.model_name,
                                transform=self.transform)

    def get_val_dataset(self):
        if self.model_name is None:
            raise ValueError("Model name must be set using `set_model_name()` before accessing the train dataset.")
        
        return PatchDataDataset(self.val_data_files, 
                                self.val_label_files, 
                                self.patch_size, 
                                shuffle=False, 
                                num_classes=self.num_classes, 
                                overlap_step=self.overlap_step,
                                target_height=self.target_height,
                                target_width=self.target_width,
                                selected_channels=self.selected_channels,
                                model_name=self.model_name,
                                transform=None)

    def get_full_dataset(self):
        """
        Get the full dataset for k-fold cross-validation.
        
        Returns:
            PatchDataDataset containing all data files
        """
        if self.model_name is None:
            raise ValueError("Model name must be set using `set_model_name()` before accessing the full dataset.")
        
        return PatchDataDataset(self.data_files, 
                                self.label_files, 
                                patch_size=self.patch_size, 
                                shuffle=self.shuffle, 
                                num_classes=self.num_classes,
                                overlap_step=self.overlap_step,
                                target_height=self.target_height,
                                target_width=self.target_width,
                                selected_channels=self.selected_channels,
                                model_name=self.model_name,
                                transform=self.transform)
    
    def set_model_name(self, model_name):
        self.model_name = model_name


def compute_pixel_confusion_matrix(preds, labels, num_classes):
    """
    Compute the confusion matrix for pixel-wise classification.
    
    Args:
        preds: Tensor of shape (N, C, H, W) (logits before softmax)
        labels: Tensor of shape (N, H, W) (ground-truth class indices)
        num_classes: Number of classes

    Returns:
        Confusion matrix (num_classes x num_classes)
    """
    if isinstance(preds, np.ndarray):
        preds = torch.tensor(preds) 

    if isinstance(labels, np.ndarray):
        labels = torch.tensor(labels)

    if preds.ndim >= 2:  
        preds = torch.argmax(preds, dim=1).cpu().numpy()
    else:
        print(f"Skipping argmax: preds has shape {preds.shape}")  
    labels = labels.cpu().numpy() 

    # Flatten 
    preds = preds.flatten()
    labels = labels.flatten()

    cm = confusion_matrix(labels, preds, labels=np.arange(num_classes))
    return cm


def compute_classification_report(preds, labels, num_classes):
    """
    Compute precision, recall, and F1-score at the pixel level.

    Args:
        preds: Tensor of shape (N, C, H, W) (logits before softmax)
        labels: Tensor of shape (N, H, W) (ground-truth class indices)
        num_classes: Number of classes

    Returns:
        Classification report (dict) with precision, recall, F1-score per class.
    """
    if isinstance(preds, np.ndarray):
        pass  
    elif isinstance(preds, torch.Tensor):
        if preds.ndim == 1:  
            preds = preds.cpu().numpy()
        elif preds.ndim == 2:  # (B, num_classes) → Classification case
            preds = torch.argmax(preds, dim=1).cpu().numpy()
        elif preds.ndim == 4:  # (B, num_classes, H, W) → Segmentation case
            preds = torch.argmax(preds, dim=1).cpu().numpy()
        else:
            raise ValueError(f"Unexpected shape for preds: {preds.shape}")
    else:
        raise TypeError(f"Unexpected type for preds: {type(preds)}")


    if isinstance(labels, np.ndarray):
        pass 
    elif isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()  
    else:
        raise TypeError(f"Unexpected type for labels: {type(labels)}")


    # Flatten 
    preds = preds.flatten()
    labels = labels.flatten()

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, labels=np.arange(num_classes), average=None)

    report = {}
    for cls in range(num_classes):
        report[f"Class {cls}"] = {
            "Precision": precision[cls],
            "Recall": recall[cls],
            "F1-Score": f1[cls]
        }
    
    return report


def visualize_feature_maps(feature_map, title, ncols=8):
    """
    Visualize each channel of a feature map.
    
    Args:
        feature_map (torch.Tensor): shape (C, H, W)
        title (str): Plot title.
        ncols (int): Number of columns for the grid.
    """
    feature_map = feature_map.numpy()  
    num_channels = feature_map.shape[0]
    nrows = int(np.ceil(num_channels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
    axes = axes.flatten()
    for i in range(num_channels):
        axes[i].imshow(feature_map[i], cmap='viridis')
        axes[i].axis('off')
        axes[i].set_title(f"Ch {i}")
    for i in range(num_channels, len(axes)):
        axes[i].axis('off')
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def create_hetero_patch_graph(img_patch):
    """
    Create a heterogeneous patch graph.
    
    Args:
        img_patch: (num_channels, patch_size, patch_size) - Can be numpy array or tensor.
        
    Returns:
        HeteroData graph.
    """
    if isinstance(img_patch, np.ndarray):
        img_patch = torch.as_tensor(img_patch, dtype=torch.float)

    num_channels, patch_size, _ = img_patch.shape
    data = HeteroData()

    # Create nodes for each channel separately
    for ch in range(num_channels):
        channel_data = img_patch[ch].reshape(-1, 1)  # Flatten
        data[f'ch_{ch}'].x = channel_data 

    # Spatial connections within each channel (4-connectivity)
    edge_list = []
    for r in range(patch_size):
        for c in range(patch_size):
            node_id = r * patch_size + c
            if c + 1 < patch_size:
                edge_list.append((node_id, node_id + 1))
                edge_list.append((node_id + 1, node_id))
            if r + 1 < patch_size:
                edge_list.append((node_id, node_id + patch_size))
                edge_list.append((node_id + patch_size, node_id))

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    for ch in range(num_channels):
        data[f'ch_{ch}', 'spatial', f'ch_{ch}'].edge_index = edge_index

    # Efficient inter-channel edges (only direct pixel-to-pixel connection)
    inter_channel_edges = torch.arange(patch_size * patch_size).repeat(2, 1)

    for src_ch in range(num_channels):
        for tgt_ch in range(num_channels):
            if src_ch != tgt_ch:
                data[f'ch_{src_ch}', 'inter_channel', f'ch_{tgt_ch}'].edge_index = inter_channel_edges

    return data


def compute_multi_image_metrics(predictions, 
                                test_label_files, 
                                num_classes, 
                                class_weights=None):
    """
    Compute comprehensive metrics for multiple image predictions.
    
    Args:
        predictions (list or np.ndarray): Predicted segmentation maps
        test_label_files (list): Paths to ground truth label files
        num_classes (int): Number of classes in segmentation
        class_weights (torch.Tensor, optional): Class weights for weighted metrics
    
    Returns:
        dict: Comprehensive metrics for all images
    """
    # Debug print to understand predictions
    print("Debug - Predictions type:", type(predictions))
    print("Debug - Predictions shape:", 
          predictions[0].shape if isinstance(predictions, list) and len(predictions) > 0 
          else "No shape")
    
    # Ensure predictions is a list
    if not isinstance(predictions, list):
        predictions = [predictions]
    
    # Initialize metrics storage
    all_metrics = {
        'per_image_metrics': {},
        'aggregated_metrics': {
            'total_pixel_confusion_matrix': np.zeros((num_classes, num_classes), dtype=int)
        }
    }
    
    # Process each image
    for idx, (pred, label_file) in enumerate(zip(predictions, test_label_files)):
        # Extract file name from path
        file_name = os.path.basename(label_file)
        # Remove file extension if needed
        file_name = os.path.splitext(file_name)[0]

        # Load ground truth
        ground_truth = np.load(label_file)
        
        # Debug print for each image
        print(f"\nDebug - Image {idx+1}:")
        print(f"Prediction shape: {pred.shape}")
        print(f"Ground truth shape: {ground_truth.shape}")
        
        # Ensure both prediction and ground truth are 2D
        if pred.ndim > 2:
            pred = pred.squeeze()
        if ground_truth.ndim > 2:
            ground_truth = ground_truth.squeeze()
        
        # Flatten for metric computation
        preds_flat = pred.flatten()
        labels_flat = ground_truth.flatten()
        
        # Ensure consistent shapes
        print(f"Flattened prediction shape: {preds_flat.shape}")
        print(f"Flattened ground truth shape: {labels_flat.shape}")
        
        # Compute confusion matrix
        pixel_cm = compute_pixel_confusion_matrix(preds_flat, labels_flat, num_classes)
        
        # Update total pixel confusion matrix
        all_metrics['aggregated_metrics']['total_pixel_confusion_matrix'] += pixel_cm
        
        # Compute Dice and IoU scores
        if class_weights is not None:
            dice, iou, dice_per_class, iou_per_class = dice_iou_score_multiclass_weighted(
                preds_flat, 
                labels_flat, 
                num_classes, 
                class_weights
            )
        else:
            # Fallback to unweighted metrics if no weights provided
            dice, iou, mcc, accuracy = None, None, None, None
            dice_per_class = np.zeros(num_classes)
            iou_per_class = np.zeros(num_classes)
        
        # Compute precision, recall, F1 scores
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels_flat, 
            preds_flat, 
            labels=range(num_classes), 
            zero_division=0
        )
        
        # Compute Matthews Correlation Coefficient
        mcc = matthews_corrcoef(labels_flat, preds_flat)
        # Compute Accuracy
        accuracy = accuracy_score(labels_flat, preds_flat)

        # Store metrics for this image
        all_metrics['per_image_metrics'][file_name] = {
            'pixel_confusion_matrix': pixel_cm.tolist(),
            'overall_metrics': {
                'dice_score': float(dice) if dice is not None else None,
                'iou_score': float(iou) if iou is not None else None,
                'mcc': float(mcc) if mcc is not None else None,
                'accuracy': float(accuracy) if accuracy is not None else None
            },
            'per_class_metrics': {
                'precision': precision.tolist(),
                'recall': recall.tolist(),
                'f1_score': f1.tolist(),
                'dice_score_per_class': dice_per_class.tolist(),
                'iou_score_per_class': iou_per_class.tolist()
            }
        }
    
    # Convert aggregated confusion matrix to list for JSON serialization
    all_metrics['aggregated_metrics']['total_pixel_confusion_matrix'] = \
        all_metrics['aggregated_metrics']['total_pixel_confusion_matrix'].tolist()
    
    return all_metrics

def compute_multi_image_metrics_ONE_MODEL(predictions, test_label_files, num_classes, class_weights=None):
    """
    Compute comprehensive metrics for multiple image predictions.
    
    Args:
        predictions (list or np.ndarray): Predicted segmentation maps
        test_label_files (list): Paths to ground truth label files
        num_classes (int): Number of classes in segmentation
        class_weights (torch.Tensor, optional): Class weights for weighted metrics
    
    Returns:
        dict: Comprehensive metrics for all images
    """
    # Debug print to understand predictions
    print("Debug - Predictions type:", type(predictions))
    print("Debug - Predictions shape:", predictions.shape if hasattr(predictions, 'shape') else "No shape")
    
    # Ensure predictions is a list
    if not isinstance(predictions, list):
        # If predictions is a single numpy array, split it based on the number of label files
        if predictions.ndim > 2:
            # Assuming first dimension is number of images
            predictions = [predictions[i] for i in range(predictions.shape[0])]
        else:
            # If single 2D array, create a list of the same prediction
            predictions = [predictions] * len(test_label_files)
    
    # Initialize metrics storage
    all_metrics = {
        'per_image_metrics': {},
        'aggregated_metrics': {
            'total_pixel_confusion_matrix': np.zeros((num_classes, num_classes), dtype=int)
        }
    }
    
    # Process each image
    for idx, (pred, label_file) in enumerate(zip(predictions, test_label_files)):
        # Load ground truth
        ground_truth = np.load(label_file)
        
        # Ensure prediction is numpy array
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        
        # Debug print for each image
        print(f"\nDebug - Image {idx+1}:")
        print(f"Prediction shape: {pred.shape}")
        print(f"Ground truth shape: {ground_truth.shape}")
        
        # Compute confusion matrix
        pixel_cm = compute_pixel_confusion_matrix(pred, ground_truth, num_classes)
        
        # Update total pixel confusion matrix
        all_metrics['aggregated_metrics']['total_pixel_confusion_matrix'] += pixel_cm
        
        # Flatten for other metric computations
        preds_flat = pred.flatten()
        labels_flat = ground_truth.flatten()
        
        # Compute Dice and IoU scores
        if class_weights is not None:
            dice, iou, dice_per_class, iou_per_class = dice_iou_score_multiclass_weighted(
                preds_flat, 
                labels_flat, 
                num_classes, 
                class_weights
            )
        
        # Compute precision, recall, F1 scores
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels_flat, 
            preds_flat, 
            labels=range(num_classes), 
            zero_division=0
        )
        
        # Store metrics for this image
        all_metrics['per_image_metrics'][f'Image_{idx+1}'] = {
            'pixel_confusion_matrix': pixel_cm.tolist(),
            'overall_metrics': {
                'dice_score': float(dice),
                'iou_score': float(iou)
            },
            'per_class_metrics': {
                'precision': precision.tolist(),
                'recall': recall.tolist(),
                'f1_score': f1.tolist(),
                'dice_score_per_class': dice_per_class.tolist(),
                'iou_score_per_class': iou_per_class.tolist()
            }
        }
    
    # Convert aggregated confusion matrix to list for JSON serialization
    all_metrics['aggregated_metrics']['total_pixel_confusion_matrix'] = \
        all_metrics['aggregated_metrics']['total_pixel_confusion_matrix'].tolist()
    
    return all_metrics

def visualize_multi_image_metrics(metrics, output_dir, class_names=None):
    """
    Create visualizations of multi-image segmentation metrics.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Use provided class names or generate default
    if class_names is None:
        num_classes = len(metrics['per_image_metrics']['Image_1']['pixel_confusion_matrix'])
        class_names = [f'Class_{i}' for i in range(num_classes)]
    
    # Visualize metrics for each image
    for img_key, img_metrics in metrics['per_image_metrics'].items():
        # Confusion Matrix Heatmap
        plt.figure(figsize=(10, 8))
        conf_matrix = np.array(img_metrics['pixel_confusion_matrix'])
        
        sns.heatmap(
            conf_matrix, 
            annot=True, 
            fmt='d', 
            cmap='Blues', 
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.title(f'Pixel-wise Confusion Matrix - {img_key}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{img_key}_confusion_matrix.png'))
        plt.close()
    
    # Aggregated Confusion Matrix
    plt.figure(figsize=(10, 8))
    total_cm = np.array(metrics['aggregated_metrics']['total_pixel_confusion_matrix'])
    sns.heatmap(
        total_cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Total Pixel-wise Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'total_confusion_matrix.png'))
    plt.close()

def format_metrics_as_table(metrics, class_names):
    """
    Convert metrics to a comprehensive, human-readable table format.
    
    Args:
        metrics (dict): Metrics dictionary
    
    Returns:
        str: Formatted table representation of metrics
    """
    # Prepare lists to store data
    table_data = []

    # Process each image's metrics
    for img_key, img_metrics in metrics['per_image_metrics'].items():
        # Overall Metrics
        overall_metrics = img_metrics['overall_metrics']
        # Per-Class Metrics
        per_class_metrics = img_metrics['per_class_metrics']
        
        # Create rows for each metric type
        for i, class_name in enumerate(class_names):
            row = {
                'Image': img_key,
                'Metric': class_name,
                'Precision': per_class_metrics['precision'][i],
                'Recall': per_class_metrics['recall'][i],
                'F1 Score': per_class_metrics['f1_score'][i],
                'Dice Score': per_class_metrics['dice_score_per_class'][i],
                'IoU Score': per_class_metrics['iou_score_per_class'][i]
            }
            table_data.append(row)
        
        # Add overall metrics row
        table_data.append({
            'Image': img_key,
            'Metric': 'Overall',
            'Precision': '-',
            'Recall': '-',
            'F1 Score': '-',
            'Dice Score': overall_metrics['dice_score'],
            'IoU Score': overall_metrics['iou_score'],
            'MCC': overall_metrics['mcc'],
            'Accuracy': overall_metrics['accuracy']
        })
    
    # Create DataFrame and format
    df = pd.DataFrame(table_data).fillna('-')
    # Format numerical values to 4 decimal places
    for col in ['Precision', 'Recall', 'F1 Score', 'Dice Score', 'IoU Score', 'MCC', 'Accuracy']:
        df[col] = df[col].apply(lambda x: f"{float(x):.4f}" if isinstance(x, (int, float)) else x)
    
    # Return formatted table
    return tabulate.tabulate(df, headers='keys', tablefmt='pretty', showindex=False)

def print_metrics_summary(metrics, class_names):
    """
    Print a detailed, formatted summary of metrics.
    
    Args:
        metrics (dict): Metrics dictionary
    """
    print("\n===== Comprehensive Metrics Summary =====")
    
    # Print per-image metrics
    for img_key, img_metrics in metrics['per_image_metrics'].items():
        print(f"\n{img_key} Metrics:")
        
        # Overall Metrics
        print("\nOverall Performance:")
        print(f"Dice Score: {img_metrics['overall_metrics']['dice_score']:.4f}")
        print(f"IoU Score: {img_metrics['overall_metrics']['iou_score']:.4f}")
        print(f"Matthews Correlation Coefficient: {img_metrics['overall_metrics']['mcc']:.4f}")
        print(f"Accuracy: {img_metrics['overall_metrics']['accuracy']:.4f}")
        
        # Per-Class Metrics
        print("\nPer-Class Performance:")
        for i, class_name in enumerate(class_names):
            print(f"\n{class_name}:")
            print(f"  Precision: {img_metrics['per_class_metrics']['precision'][i]:.4f}")
            print(f"  Recall: {img_metrics['per_class_metrics']['recall'][i]:.4f}")
            print(f"  F1 Score: {img_metrics['per_class_metrics']['f1_score'][i]:.4f}")
            print(f"  Dice Score: {img_metrics['per_class_metrics']['dice_score_per_class'][i]:.4f}")
            print(f"  IoU Score: {img_metrics['per_class_metrics']['iou_score_per_class'][i]:.4f}")
    
    # Aggregated Confusion Matrix
    print("\n===== Total Pixel-wise Confusion Matrix =====")
    print(pd.DataFrame(
        metrics['aggregated_metrics']['total_pixel_confusion_matrix'], 
        columns=class_names, 
        index=class_names
    ))


def save_model_metrics(model_name, model_metrics, output_dir, class_names):
    """
    Save model metrics to text files.
    
    Args:
        model_name (str): Name of the model
        model_metrics (dict): Metrics dictionary
        output_dir (str): Directory to save metrics files
        class_names (list): List of class names for reporting
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare filenames
    table_metrics_filename = os.path.join(output_dir, f"{model_name}_metrics_table.txt")
    detailed_metrics_filename = os.path.join(output_dir, f"{model_name}_metrics_summary.txt")
    json_metrics_filename = os.path.join(output_dir, f"{model_name}_metrics.json")
    
    # Save tabulated metrics
    try:
        table_output = format_metrics_as_table(model_metrics, class_names)
        with open(table_metrics_filename, 'w') as f:
            f.write(table_output)
        print(f"Saved tabulated metrics to {table_metrics_filename}")
    except Exception as e:
        print(f"Error saving tabulated metrics: {e}")
    
    # Save detailed metrics summary
    try:
        # Capture print_metrics_summary output
        from io import StringIO
        import sys
        
        # Redirect stdout to a string buffer
        old_stdout = sys.stdout
        result = StringIO()
        sys.stdout = result
        
        # Call print_metrics_summary
        print_metrics_summary(model_metrics, class_names)
        
        # Restore stdout
        sys.stdout = old_stdout
        
        # Get the captured output
        detailed_metrics = result.getvalue()
        
        # Write to file
        with open(detailed_metrics_filename, 'w') as f:
            f.write(detailed_metrics)
        print(f"Saved detailed metrics summary to {detailed_metrics_filename}")
    except Exception as e:
        print(f"Error saving detailed metrics summary: {e}")
    
    # Save full metrics as JSON for comprehensive reference
    try:
        with open(json_metrics_filename, 'w') as f:
            json.dump(model_metrics, f, indent=2)
        print(f"Saved full metrics JSON to {json_metrics_filename}")
    except Exception as e:
        print(f"Error saving JSON metrics: {e}")



def create_individual_plots(history, model_name, outdir):
    """
    Create plots for an individual model.
    
    Args:
        history: Training history dictionary
        model_name: Name of the model
    """
    fig1, axes1 = plt.subplots(1, 5, figsize=(18, 5))
    
    # Plot Loss
    axes1[0].plot(history['train_losses'], label='Train Loss', marker='o')
    axes1[0].plot(history['val_losses'], label='Validation Loss', marker='o')
    axes1[0].set_xlabel("Epoch")
    axes1[0].set_ylabel("Loss")
    axes1[0].set_title("Loss Curves")
    axes1[0].legend()
    
    axes1[1].plot(history['train_ious'], label='Train IoU', marker='o')
    axes1[1].plot(history['val_ious'], label='Validation IoU', marker='o')
    axes1[1].set_xlabel("Epoch")
    axes1[1].set_ylabel("IoU")
    axes1[1].set_title("IoU Curves")
    axes1[1].legend()

    #train_dices_cpu = [dice.cpu().numpy() for dice in train_dices]
    #val_dices_cpu = [dice.cpu().numpy() for dice in val_dices]
    train_dices_cpu = [dice for dice in history['train_dices']]
    val_dices_cpu = [dice for dice in history['val_dices']]
    axes1[2].plot(train_dices_cpu, label='Train Dice', marker='o')
    axes1[2].plot(val_dices_cpu, label='Validation Dice', marker='o')
    axes1[2].set_xlabel("Epoch")
    axes1[2].set_ylabel("Dice Score")
    axes1[2].set_title("Dice Score Curves")
    axes1[2].legend()

    # Plot MCC
    axes1[3].plot(history['train_mcc'], label='Train MCC', marker='o')
    axes1[3].plot(history['val_mcc'], label='Validation MCC', marker='o')
    axes1[3].set_xlabel("Epoch")
    axes1[3].set_ylabel("MCC")
    axes1[3].set_title("Matthews Correlation Coefficient")
    axes1[3].legend()
    
    # Plot Accuracy
    axes1[4].plot(history['train_accuracy'], label='Train Accuracy', marker='o')
    axes1[4].plot(history['val_accuracy'], label='Validation Accuracy', marker='o')
    axes1[4].set_xlabel("Epoch")
    axes1[4].set_ylabel("Accuracy")
    axes1[4].set_title("Accuracy Curves")
    axes1[4].legend()
   
    plt.tight_layout()
    plt.savefig(outdir + f"{model_name}_metrics.png", dpi=300)
    plt.close()

def create_comparison_plots(all_models_history, class_names, outdir):
    """
    Create plots comparing metrics across different models with consistent colors
    for train/validation pairs.
    
    Args:
        all_models_history: Dictionary mapping model names to their training histories
    """
    # Define a color cycle to ensure consistent colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Plot Loss, IoU, Dice comparison
    fig1, axes1 = plt.subplots(1, 5, figsize=(24, 6))
    
    # Plot Loss
    for i, (model_name, history) in enumerate(all_models_history.items()):
        color = colors[i % len(colors)]  
        axes1[0].plot(history['train_losses'], label=f'{model_name} Train', 
                     marker='o', linestyle='-', color=color)
        axes1[0].plot(history['val_losses'], label=f'{model_name} Val', 
                     marker='x', linestyle='--', color=color)
    
    axes1[0].set_xlabel("Epoch")
    axes1[0].set_ylabel("Loss")
    axes1[0].set_title("Loss Comparison")
    axes1[0].legend()
    
    # Plot IoU
    for i, (model_name, history) in enumerate(all_models_history.items()):
        color = colors[i % len(colors)]  
        axes1[1].plot(history['train_ious'], label=f'{model_name} Train', 
                     marker='o', linestyle='-', color=color)
        axes1[1].plot(history['val_ious'], label=f'{model_name} Val', 
                     marker='x', linestyle='--', color=color)
    
    axes1[1].set_xlabel("Epoch")
    axes1[1].set_ylabel("IoU")
    axes1[1].set_title("IoU Comparison")
    axes1[1].legend()
    
    # Plot Dice
    for i, (model_name, history) in enumerate(all_models_history.items()):
        color = colors[i % len(colors)]  
        train_dices = [dice for dice in history['train_dices']]
        val_dices = [dice for dice in history['val_dices']]
        axes1[2].plot(train_dices, label=f'{model_name} Train', 
                     marker='o', linestyle='-', color=color)
        axes1[2].plot(val_dices, label=f'{model_name} Val', 
                     marker='x', linestyle='--', color=color)
    
    axes1[2].set_xlabel("Epoch")
    axes1[2].set_ylabel("Dice Score")
    axes1[2].set_title("Dice Score Comparison")
    axes1[2].legend()
    
    # Plot MCC
    for i, (model_name, history) in enumerate(all_models_history.items()):
        if 'train_mcc' in history and 'val_mcc' in history:
            color = colors[i % len(colors)]
            axes1[3].plot(history['train_mcc'], label=f'{model_name} Train', 
                         marker='o', linestyle='-', color=color)
            axes1[3].plot(history['val_mcc'], label=f'{model_name} Val', 
                         marker='x', linestyle='--', color=color)
    
    axes1[3].set_xlabel("Epoch")
    axes1[3].set_ylabel("MCC")
    axes1[3].set_title("Matthews Correlation Coefficient")
    axes1[3].legend()
    
    # Plot Accuracy
    for i, (model_name, history) in enumerate(all_models_history.items()):
        if 'train_accuracy' in history and 'val_accuracy' in history:
            color = colors[i % len(colors)]
            axes1[4].plot(history['train_accuracy'], label=f'{model_name} Train', 
                         marker='o', linestyle='-', color=color)
            axes1[4].plot(history['val_accuracy'], label=f'{model_name} Val', 
                         marker='x', linestyle='--', color=color)
    
    axes1[4].set_xlabel("Epoch")
    axes1[4].set_ylabel("Accuracy")
    axes1[4].set_title("Accuracy Comparison")
    axes1[4].legend()
    
    plt.tight_layout()
    plt.savefig(outdir + "model_comparison_metrics.png", dpi=300)
    plt.close()
    
    # Create class-specific comparison plots
    for cls in range(NB_CLASSES):
        fig, axes = plt.subplots(2, 3, figsize=(24, 10))
        
        # Plot Precision
        for i, (model_name, history) in enumerate(all_models_history.items()):
            color = colors[i % len(colors)]  
            train_prec = [epoch_arr[cls] for epoch_arr in history['train_precision_history']]
            val_prec = [epoch_arr[cls] for epoch_arr in history['val_precision_history']]
            axes[0, 0].plot(train_prec, label=f'{model_name} Train', 
                           marker='o', linestyle='-', color=color)
            axes[0, 0].plot(val_prec, label=f'{model_name} Val', 
                           marker='x', linestyle='--', color=color)
        
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Precision")
        axes[0, 0].set_title(class_names[cls] + " Precision")
        axes[0, 0].legend()
        
        # Plot Recall
        for i, (model_name, history) in enumerate(all_models_history.items()):
            color = colors[i % len(colors)]  
            train_rec = [epoch_arr[cls] for epoch_arr in history['train_recall_history']]
            val_rec = [epoch_arr[cls] for epoch_arr in history['val_recall_history']]
            axes[0, 1].plot(train_rec, label=f'{model_name} Train', 
                           marker='o', linestyle='-', color=color)
            axes[0, 1].plot(val_rec, label=f'{model_name} Val', 
                           marker='x', linestyle='--', color=color)
        
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Recall")
        axes[0, 1].set_title(class_names[cls] + " Recall")
        axes[0, 1].legend()
        
        # Plot F1
        for i, (model_name, history) in enumerate(all_models_history.items()):
            color = colors[i % len(colors)] 
            train_f1 = [epoch_arr[cls] for epoch_arr in history['train_f1_history']]
            val_f1 = [epoch_arr[cls] for epoch_arr in history['val_f1_history']]
            axes[0, 2].plot(train_f1, label=f'{model_name} Train', 
                           marker='o', linestyle='-', color=color)
            axes[0, 2].plot(val_f1, label=f'{model_name} Val', 
                           marker='x', linestyle='--', color=color)
        
        axes[0, 2].set_xlabel("Epoch")
        axes[0, 2].set_ylabel("F1 Score")
        axes[0, 2].set_title(class_names[cls] + " F1 Score")
        axes[0, 2].legend()
        
        # Plot class-specific Dice
        for i, (model_name, history) in enumerate(all_models_history.items()):
            color = colors[i % len(colors)]  
            try:
                train_dice = [epoch_arr[cls].item() if hasattr(epoch_arr[cls], 'item') else epoch_arr[cls] 
                             for epoch_arr in history['train_dice_per_class']]
                val_dice = [epoch_arr[cls].item() if hasattr(epoch_arr[cls], 'item') else epoch_arr[cls] 
                           for epoch_arr in history['val_dice_per_class']]
            except (KeyError, IndexError) as e:
                print(f"Warning: Could not extract per-class Dice for {model_name}, {class_names[cls]}: {e}")
                train_dice = history['train_dices']
                val_dice = history['val_dices']
                
            epochs = range(len(train_dice))
            axes[1, 0].plot(epochs, train_dice, label=f'{model_name} Train', 
                           marker='o', linestyle='-', color=color)
            axes[1, 0].plot(epochs, val_dice, label=f'{model_name} Val', 
                           marker='x', linestyle='--', color=color)

        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Dice")
        axes[1, 0].set_title(class_names[cls] + " Dice")
        axes[1, 0].legend()

        # Plot class-specific IoU
        for i, (model_name, history) in enumerate(all_models_history.items()):
            color = colors[i % len(colors)] 
            try:
                train_iou = [epoch_arr[cls].item() if hasattr(epoch_arr[cls], 'item') else epoch_arr[cls] 
                             for epoch_arr in history['train_iou_per_class']]
                val_iou = [epoch_arr[cls].item() if hasattr(epoch_arr[cls], 'item') else epoch_arr[cls] 
                           for epoch_arr in history['val_iou_per_class']]
            except (KeyError, IndexError) as e:
                print(f"Warning: Could not extract per-class IoU for {model_name}, {class_names[cls]}: {e}")
                train_iou = history['train_ious']
                val_iou = history['val_ious']
                
            epochs = range(len(train_iou))
            axes[1, 1].plot(epochs, train_iou, label=f'{model_name} Train', 
                           marker='o', linestyle='-', color=color)
            axes[1, 1].plot(epochs, val_iou, label=f'{model_name} Val', 
                           marker='x', linestyle='--', color=color)

        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("IoU")
        axes[1, 1].set_title(class_names[cls] + " IoU")
        axes[1, 1].legend()
        
        # Add empty plot or custom plot in the remaining panel
        axes[1, 2].set_visible(False)  # Hide the unused subplot
        # Alternatively, you could add another metric here
        
        plt.tight_layout()
        plt.savefig(outdir + "model_comparison_class_" + class_names[cls] + ".png", dpi=300)
        plt.close()

def visualize_spectral_signatures(patch, normalized_patch, row=10, col=10):
    """
    Visualize spectral signatures before and after normalization at a specific pixel location
    
    Args:
        patch: Original hyperspectral patch [C, H, W]
        normalized_patch: Normalized hyperspectral patch [C, H, W]
        row, col: Pixel location to visualize
    """   
    # Get the spectral signature at the specified pixel location
    original_sig = patch[:, row, col]
    normalized_sig = normalized_patch[:, row, col]
    
    bands = range(len(original_sig))
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(bands, original_sig.numpy(), 'b-', linewidth=2)
    plt.title('Original Spectral Signature')
    plt.xlabel('Spectral Band')
    plt.ylabel('Intensity')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(bands, normalized_sig.numpy(), 'r-', linewidth=2)
    plt.title('Pixel-Normalized Spectral Signature')
    plt.xlabel('Spectral Band')
    plt.ylabel('Normalized Intensity')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + 'spectral_comparison.png')
    plt.close()
    
def normalize_spectral_signature(img_tensor):
    """
    Normalize each pixel's spectral signature independently using min-max normalization.
    Works with both batched input [B, C, H, W] and single image [C, H, W].
    
    Args:
        img_tensor: PyTorch tensor of shape [B, C, H, W] or [C, H, W]
        
    Returns:
        Normalized tensor of the same shape
    """
    original_shape = img_tensor.shape
    
    if len(original_shape) == 4:  # Batched input [B, C, H, W]
        B, C, H, W = original_shape
        # Reshape to [B*H*W, C] to process all pixels at once
        img_reshaped = img_tensor.permute(0, 2, 3, 1).reshape(-1, C)
        
        # Get min and max for each pixel's spectral signature
        pixel_min, _ = torch.min(img_reshaped, dim=1, keepdim=True)
        pixel_max, _ = torch.max(img_reshaped, dim=1, keepdim=True)
        
        # Handle the case where min equals max (flat spectral signature)
        denominator = pixel_max - pixel_min
        denominator = torch.where(denominator < 1e-4, torch.tensor(1e-4).to(denominator.device), denominator)
        
        # Apply min-max normalization
        normalized = (img_reshaped - pixel_min) / denominator
        
        # Reshape back to original dimensions
        return normalized.reshape(B, H, W, C).permute(0, 3, 1, 2)
        
    else:  # Non-batched input [C, H, W]
        C, H, W = original_shape
        # Reshape to [H*W, C] to process all pixels at once
        img_reshaped = img_tensor.permute(1, 2, 0).reshape(-1, C)
        
        # Get min and max for each pixel's spectral signature
        pixel_min, _ = torch.min(img_reshaped, dim=1, keepdim=True)
        pixel_max, _ = torch.max(img_reshaped, dim=1, keepdim=True)
        
        # Handle the case where min equals max (flat spectral signature)
        denominator = pixel_max - pixel_min
        denominator = torch.where(denominator < 1e-4, torch.tensor(1e-4).to(denominator.device), denominator)
        
        # Apply min-max normalization
        normalized = (img_reshaped - pixel_min) / denominator
        
        # Reshape back to original dimensions
        return normalized.reshape(H, W, C).permute(2, 0, 1)

def calculate_class_weights(label_files, num_classes):
    """
    Calculate class weights based on pixel frequency in the dataset.
    
    Args:
        label_files: List of paths to ground truth label files
        num_classes: Number of classes in the segmentation task
        
    Returns:
        Class weights as a numpy array
    """
    # Initialize pixel count per class
    class_counts = np.zeros(num_classes, dtype=np.int64)
    
    # Count pixels for each class
    print("Counting pixels per class...")
    for label_file in tqdm(label_files):
        # Load label
        label = np.load(label_file)
        # Count pixels per class in this image
        for class_idx in range(num_classes):
            class_counts[class_idx] += np.sum(label == class_idx)
    
    print("Pixel counts per class:", class_counts)
    
    # Calculate weights (inverse frequency)
    class_weights = 1.0 / class_counts
    # Normalize weights
    # class_weights = class_weights / class_weights.sum()
    # Alternative: 
    #class_weights = class_weights / class_weights.min()  # Relative to most frequent class
    class_weights = 1.0 / np.log1p(class_counts)  # Softer scaling
    # class_weights = class_weights / class_weights.min() * 2  # Boost magnitude
    # class_weights = 1.0 / np.sqrt(class_counts) #less aggressive weighting
    
    return class_weights
