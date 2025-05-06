import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
   
class WeightedDiceLoss(nn.Module):
    def __init__(self, class_weights, smooth=1e-5):
        """
        Args:
            class_weights (torch.Tensor): Weight for each class (shape: [num_classes]).
            smooth (float): Smoothing factor to avoid division by zero.
        """
        super(WeightedDiceLoss, self).__init__()
        self.class_weights = class_weights
        self.smooth = smooth

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Predicted logits (shape: [N, C, H, W]).
            targets (torch.Tensor): Ground truth class indices (shape: [N, H, W]).
        
        Returns:
            torch.Tensor: Weighted Dice loss.
        """
        # Convert logits to probabilities (if necessary)
        inputs = F.softmax(inputs, dim=1)  # Shape: [N, C, H, W]

        # If the network output spatial size is different from the target, interpolate.
        target_size = targets.shape[1:]  # expecting targets shape [N, H, W]
        if inputs.shape[2:] != target_size:
            inputs = F.interpolate(inputs, size=target_size, mode='bilinear', align_corners=False)
    

        # Convert targets to one-hot encoding
        num_classes = inputs.shape[1]
        targets = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()  # Shape: [N, C, H, W]

        # Compute Dice score for each class
        dice_scores = []
        eps = 1e-6  # Small value to avoid division by zero
        for class_idx in range(num_classes):
            input_class = inputs[:, class_idx, ...]  # Predicted probabilities for class_idx
            target_class = targets[:, class_idx, ...]  # Ground truth mask for class_idx

            # Compute intersection and union
            intersection = (input_class * target_class).sum()
            union = input_class.sum() + target_class.sum()
            
            # Compute Dice score for this class
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth + eps)
            dice_scores.append(dice)

        # Stack Dice scores into a tensor (preserves gradients)
        dice_scores = torch.stack(dice_scores)

        # Compute weighted Dice score
        weighted_dice = (dice_scores * self.class_weights).sum() / self.class_weights.sum()

        # Ensure valid loss value (in range [0, 1])
        #weighted_dice = torch.clamp(weighted_dice, min=eps, max=1.0)

        # Return Dice loss
        return 1.0 - weighted_dice
    

class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma, class_weights=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights  # Tensor of shape [num_classes]

        if class_weights is not None:
            if not isinstance(class_weights, torch.Tensor):
                raise ValueError("class_weights must be a torch.Tensor")
            self.class_weights = class_weights.to(dtype=torch.float32)

    def forward(self, inputs, targets):
        # Compute the cross-entropy loss without reduction
        if self.class_weights is None:
            BCE_loss = F.cross_entropy(inputs, targets)
        else:
            BCE_loss = F.cross_entropy(inputs, targets, weight=self.class_weights)

        # Compute pt (probability of the true class)
        pt = torch.exp(-BCE_loss)  # pt is e^(-BCE_loss), where BCE_loss is the negative log-likelihood

        # Apply focal loss weighting
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        return F_loss


def mean_iou(preds, labels, num_classes):
    """
    Compute Mean IoU (Intersection over Union) for multi-class segmentation.
    Args:
        preds: Tensor of shape (N, H_pred, W_pred) (predicted class indices)
        labels: Tensor of shape (N, H, W) (ground-truth class indices)
        num_classes: Number of classes
    Returns:
        Mean IoU score (float)
    """
    iou_list = []
    
    for cls in range(num_classes):
        pred_cls = (preds == cls)
        label_cls = (labels == cls)

        intersection = (pred_cls & label_cls).sum().item()
        union = (pred_cls | label_cls).sum().item()

        if union == 0:  # Avoid division by zero
            iou = float('nan')  # Ignore this class
        else:
            iou = intersection / union
        
        iou_list.append(iou)

    return np.nanmean(iou_list)  # Mean IoU across all classes


def dice_score(preds, labels, num_classes, eps=1e-8):
    """
    Compute Dice Score (F1 Score for segmentation).
    Args:
        preds: Tensor of shape (N, H_pred, W_pred) (predicted class indices)
        labels: Tensor of shape (N, H, W) (ground-truth class indices)
        num_classes: Number of classes
        eps: Small value to avoid division by zero
    Returns:
        Mean Dice score (float)
    """
    dice_list = []
    
    for cls in range(num_classes):
        pred_cls = (preds == cls)
        label_cls = (labels == cls)

        intersection = (pred_cls & label_cls).sum().item()
        dice = (2. * intersection) / (pred_cls.sum().item() + label_cls.sum().item() + eps)  # Avoid division by zero
        
        dice_list.append(dice)

    return np.nanmean(dice_list)  # Mean Dice across all classes

def dice_score_multiclass_weighted(y_pred, y_true, num_classes, class_weights, smooth=1e-6):
    # Ensure the inputs are torch tensors
    if not isinstance(y_pred, torch.Tensor):
        y_pred = torch.as_tensor(y_pred)
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.as_tensor(y_true)

    # Initialize Dice scores for each class
    dice_scores = []
    
    for class_idx in range(num_classes):
        # Create binary masks for the current class
        y_pred_class = (y_pred == class_idx).float()  # Predicted mask for class_idx
        y_true_class = (y_true == class_idx).float()  # Ground truth mask for class_idx
        
        # Compute intersection and union sums
        intersection = (y_pred_class * y_true_class).sum()
        union = y_pred_class.sum() + y_true_class.sum()
        
        # Compute Dice score for this class
        dice = (2.0 * intersection + smooth) / (union + smooth)
        dice_scores.append(dice)
    
    # Stack and compute weighted Dice score
    dice_scores = torch.stack(dice_scores)
    class_weights = torch.as_tensor(class_weights).float().to(dice_scores.device)
    weighted_dice = (dice_scores * class_weights).sum() / class_weights.sum()
    return weighted_dice.item()

def dice_iou_score_multiclass_weighted(y_pred, y_true, num_classes, class_weights=None, smooth=1e-6):
    if not isinstance(y_pred, torch.Tensor):
        y_pred = torch.as_tensor(y_pred)
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.as_tensor(y_true)

    # Initialize Dice and IoU scores for each class
    dice_scores = []
    iou_scores = []
    
    for class_idx in range(num_classes):
        # Create binary masks for the current class
        y_pred_class = (y_pred == class_idx).float()
        y_true_class = (y_true == class_idx).float()
        
        # Intersection and union
        intersection = (y_pred_class * y_true_class).sum()
        union = y_pred_class.sum() + y_true_class.sum()
        union_iou = (y_pred_class + y_true_class).sum() - intersection
        
        # Dice Score: 2 * intersection / (sum of individual areas)
        dice = (2.0 * intersection + smooth) / (union + smooth)
        dice_scores.append(dice)
        
        # IoU Score: intersection / union
        iou = (intersection + smooth) / (union_iou + smooth)
        iou_scores.append(iou)
    
    # Stack the scores into tensors
    dice_scores = torch.stack(dice_scores)
    iou_scores = torch.stack(iou_scores)
    
    if class_weights is not None:
        class_weights = torch.as_tensor(class_weights).float().to(dice_scores.device)
    else:
        # Use equal weights if class_weights is None
        class_weights = torch.ones(num_classes, device=dice_scores.device)
    
    # Compute weighted mean Dice and IoU
    weighted_dice = (dice_scores * class_weights).sum() / class_weights.sum()
    weighted_iou = (iou_scores * class_weights).sum() / class_weights.sum()
    
    # Return both weighted and per-class values
    return (
        weighted_dice.item(), weighted_iou.item(), 
        dice_scores.cpu().numpy(), iou_scores.cpu().numpy()
    )

def compute_pixel_confusion_matrix(preds, labels, num_classes):
    """
    Compute the confusion matrix for pixel-wise classification.
    
    Args:
        preds: Tensor or array of predictions 
        labels: Tensor or array of ground truth class indices
        num_classes: Number of classes
    
    Returns:
        Confusion matrix (num_classes x num_classes)
    """
    # Convert to numpy if needed
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    # Handle multi-dimensional inputs
    if preds.ndim > 2:
        # If input is logits or multi-dimensional, apply argmax
        if preds.ndim == 4:  # Shape: (N, C, H, W)
            preds = np.argmax(preds, axis=1)
        elif preds.ndim == 3:  # Shape: (N, H, W)
            preds = preds[0] if preds.shape[0] > 1 else preds.squeeze()
    
    # Ensure inputs are 1D
    preds = preds.ravel()
    labels = labels.ravel()
    
    # Verify input
    assert preds.shape == labels.shape, f"Prediction and label shapes must match. Got {preds.shape} and {labels.shape}"
    
    # Validate class range
    assert np.min(preds) >= 0 and np.max(preds) < num_classes, \
        f"Predictions must be in range [0, {num_classes-1}]. Got range [{np.min(preds)}, {np.max(preds)}]"
    assert np.min(labels) >= 0 and np.max(labels) < num_classes, \
        f"Labels must be in range [0, {num_classes-1}]. Got range [{np.min(labels)}, {np.max(labels)}]"
    
    # Compute confusion matrix
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(labels, preds):
        cm[t, p] += 1
    
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


    # Flatten for pixel-wise comparison
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

# Multi-task loss function for training
def multi_task_loss(outputs, targets, class_weights=None):
    # Main segmentation loss (using your existing weighted loss)
    if class_weights is not None:
        main_loss = WeightedDiceLoss(class_weights)(outputs['main_output'], targets)
    else:
        main_loss = F.cross_entropy(outputs['main_output'], targets)
    
    # Auxiliary segmentation losses
    aux_loss1 = F.cross_entropy(outputs['aux_output1'], targets)
    aux_loss2 = F.cross_entropy(outputs['aux_output2'], targets)
    aux_loss3 = F.cross_entropy(outputs['aux_output3'], targets)
    
    # Contrastive loss implementation
    # This is a simplified example - you'll need to customize based on your exact needs
    def contrastive_learning_loss(features, temperature=0.5):
        # Normalize features to unit hypersphere
        features = F.normalize(features, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / temperature
        
        # Exclude the diagonal (self-similarity)
        mask = torch.eye(similarity_matrix.shape[0], device=similarity_matrix.device)
        mask = 1 - mask
        
        # Compute contrastive loss
        exp_sim = torch.exp(similarity_matrix) * mask
        loss = -torch.log(exp_sim / exp_sim.sum(dim=1, keepdim=True) + 1e-8).mean()
        
        return loss
    
    contrastive_loss = contrastive_learning_loss(outputs['contrastive_feat'])
    
    # Combine losses with weights
    total_loss = (
        main_loss + 
        0.3 * aux_loss1 + 
        0.2 * aux_loss2 + 
        0.2 * aux_loss3 + 
        0.1 * contrastive_loss
    )
    
    # You could also return individual losses for monitoring
    loss_dict = {
        'main_loss': main_loss.item(),
        'aux_loss1': aux_loss1.item(),
        'aux_loss2': aux_loss2.item(),
        'aux_loss3': aux_loss3.item(),
        'contrastive_loss': contrastive_loss.item(),
        'total_loss': total_loss.item()
    }
    
    return total_loss, loss_dict
