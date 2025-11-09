import torch.nn as nn
import torch.nn.functional as F
import torch 

# =========================== Define your custom loss function ===========================================

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in semantic segmentation.
    
    Reference: 
    Lin et al. "Focal Loss for Dense Object Detection" (ICCV 2017)
    https://arxiv.org/abs/1708.02002
    
    The Focal Loss is designed to address class imbalance by down-weighting 
    the loss assigned to well-classified examples (easy examples) and focusing 
    training on hard examples.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    where:
    - p_t is the model's estimated probability for the correct class
    - gamma (focusing parameter): increases loss for hard examples
    - alpha (balancing parameter): balances class weights
    
    Args:
        alpha: Weighting factor in [0, 1] to balance positive/negative examples.
               Can be a scalar or list of weights per class. Default: 0.25
        gamma: Focusing parameter >= 0. gamma=0 is equivalent to CrossEntropyLoss.
               Higher gamma focuses more on hard examples. Default: 2.0
        ignore_index: Specifies a target value that is ignored. Default: 255
        reduction: Specifies the reduction to apply to output. Default: 'mean'
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, ignore_index=255, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N, C, H, W) raw predictions (logits) from model
            targets: (N, H, W) ground truth labels
        
        Returns:
            Focal loss value
        """
        # Get cross entropy loss without reduction
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', 
                                   ignore_index=self.ignore_index)
        
        # Get probabilities from logits
        p = F.softmax(inputs, dim=1)  # (N, C, H, W)
        
        # Get the probability of the true class for each pixel
        # Gather the probabilities at the target indices
        N, C, H, W = inputs.shape
        targets_one_hot = F.one_hot(targets, num_classes=C)  # (N, H, W, C)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # (N, C, H, W)
        
        # Handle ignore_index
        valid_mask = (targets != self.ignore_index).float()  # (N, H, W)
        
        # Get probability of true class: p_t
        p_t = (p * targets_one_hot).sum(dim=1)  # (N, H, W)
        
        # Calculate focal weight: (1 - p_t)^gamma
        # This down-weights easy examples (high p_t) and focuses on hard ones (low p_t)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply focal weight to cross entropy loss
        focal_loss = focal_weight * ce_loss
        
        # Apply alpha weighting (optional, can be per-class)
        if self.alpha is not None:
            focal_loss = self.alpha * focal_loss
        
        # Apply ignore mask
        focal_loss = focal_loss * valid_mask
        
        # Reduce loss
        if self.reduction == 'mean':
            return focal_loss.sum() / (valid_mask.sum() + 1e-6)
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedFocalLoss(nn.Module):
    """
    Focal Loss with per-class weights to handle class imbalance.
    
    This extends FocalLoss by adding per-class weighting based on class frequency.
    Rare classes get higher weights, common classes get lower weights.
    
    Args:
        alpha: Per-class weights (list or tensor of shape [num_classes])
        gamma: Focusing parameter. Default: 2.0
        ignore_index: Specifies target value to ignore. Default: 255
        reduction: Reduction method. Default: 'mean'
    """
    
    def __init__(self, alpha=None, gamma=2.0, ignore_index=255, reduction='mean'):
        super(WeightedFocalLoss, self).__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
        
        # Default weights for Pascal VOC (21 classes including background)
        # These are inverse frequency weights computed from training set
        if alpha is None:
            # You can compute these from your dataset or use uniform weights
            alpha = [1.0] * 21  # Uniform weights as default
        
        self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float32))
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N, C, H, W) logits
            targets: (N, H, W) labels
        """
        N, C, H, W = inputs.shape
        
        # Compute cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none',
                                   ignore_index=self.ignore_index)
        
        # Get softmax probabilities
        p = F.softmax(inputs, dim=1)  # (N, C, H, W)
        
        # Gather probabilities at target locations
        targets_expanded = targets.unsqueeze(1)  # (N, 1, H, W)
        p_t = p.gather(1, targets_expanded.clamp(0, C-1)).squeeze(1)  # (N, H, W)
        
        # Handle ignore_index
        valid_mask = (targets != self.ignore_index)
        p_t = torch.where(valid_mask, p_t, torch.ones_like(p_t))
        
        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        
        # Get per-class alpha weights
        alpha_t = self.alpha[targets.clamp(0, C-1)]  # (N, H, W)
        alpha_t = torch.where(valid_mask, alpha_t, torch.zeros_like(alpha_t))
        
        # Combine everything
        focal_loss = alpha_t * focal_weight * ce_loss
        
        # Reduce
        if self.reduction == 'mean':
            return focal_loss.sum() / (valid_mask.sum().float() + 1e-6)
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DiceLoss(nn.Module):
    """
    Dice Loss for semantic segmentation.
    
    Dice loss focuses on the overlap between prediction and ground truth,
    making it robust to class imbalance. It's commonly used in medical 
    image segmentation.
    
    Dice = 2 * |X ∩ Y| / (|X| + |Y|)
    DiceLoss = 1 - Dice
    
    Args:
        smooth: Smoothing constant to avoid division by zero. Default: 1.0
        ignore_index: Target value to ignore. Default: 255
    """
    
    def __init__(self, smooth=1.0, ignore_index=255):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N, C, H, W) logits
            targets: (N, H, W) labels
        """
        N, C, H, W = inputs.shape
        
        # Get probabilities
        probs = F.softmax(inputs, dim=1)  # (N, C, H, W)
        
        # Convert targets to one-hot
        targets_one_hot = F.one_hot(targets.clamp(0, C-1), num_classes=C)  # (N, H, W, C)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # (N, C, H, W)
        
        # Create mask for valid pixels
        valid_mask = (targets != self.ignore_index).unsqueeze(1).float()  # (N, 1, H, W)
        
        # Apply mask
        probs = probs * valid_mask
        targets_one_hot = targets_one_hot * valid_mask
        
        # Compute Dice coefficient per class
        intersection = (probs * targets_one_hot).sum(dim=(2, 3))  # (N, C)
        union = probs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))  # (N, C)
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)  # (N, C)
        
        # Dice loss
        dice_loss = 1 - dice.mean()
        
        return dice_loss


class CombinedLoss(nn.Module):
    """
    Combined Cross Entropy + Dice Loss for semantic segmentation.
    
    This combines the benefits of both losses:
    - CrossEntropy: Good for pixel-wise classification
    - Dice: Good for handling class imbalance and overlap
    
    Args:
        ce_weight: Weight for cross entropy loss. Default: 0.5
        dice_weight: Weight for dice loss. Default: 0.5
        ignore_index: Target value to ignore. Default: 255
    """
    
    def __init__(self, ce_weight=0.5, dice_weight=0.5, ignore_index=255):
        super(CombinedLoss, self).__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice_loss = DiceLoss(ignore_index=ignore_index)
        
    def forward(self, inputs, targets):
        ce = self.ce_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        return self.ce_weight * ce + self.dice_weight * dice
