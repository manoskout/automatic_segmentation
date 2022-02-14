import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

# SR : Segmentation Result
# GT : Ground Truth

def get_accuracy(SR,GT,threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)
    corr = torch.sum(SR==GT)
    tensor_size = SR.size(0)*SR.size(1)*SR.size(2)*SR.size(3)
    acc = float(corr)/float(tensor_size)

    return acc

def get_specificity(SR,GT,threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)

    # TN : True Negative
    # FP : False Positive
    TN = ((SR==0)+(GT==0))==2
    FP = ((SR==1)+(GT==0))==2

    SP = float(torch.sum(TN))/(float(torch.sum(TN+FP)) + 1e-6)
    
    return SP

def recall(inputs: Tensor, targets: Tensor) -> float:
    """
    """
    tp = (inputs * targets).sum().to(torch.float32)
    fn = (inputs * (1 - targets)).sum().to(torch.float32)
    epsilon = 1e-7
    return tp / (tp + fn + epsilon)

def precision(inputs: Tensor, targets: Tensor) -> float:
    """
    """
    tp = (inputs * targets).sum().to(torch.float32)
    fp = ((1 - inputs) * targets).sum().to(torch.float32)
    epsilon = 1e-7
    return tp / (tp+fp + epsilon)

def fi_score(targets:torch.Tensor, inputs:torch.Tensor) -> torch.Tensor:
    '''Calculate F1 score. Can work with gpu tensors
    
    The original implmentation is written by Michal Haltuf on Kaggle.
    
    Returns
    -------
    torch.Tensor
        `ndim` == 1. 0 <= val <= 1
    
    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
    
    '''
    # assert inputs.ndim == 1
    # assert targets.ndim == 1 or targets.ndim == 2
    
    # if targets.ndim == 2:
    #     targets = targets.argmax(dim=1)
        
    
    # precision = precision(inputs, targets)
    # recall = recall(inputs,targets)
    # epsilon = 1e-7

    # f1 = 2* (precision*recall) / (precision + recall + epsilon)
    f1 =0
    return f1


def jaccard_score(preds, inputs):
    inputs = inputs.view(-1)
    preds = preds.view(-1)

    # Ignore IoU for background class ("0")
    # for cls in xrange(1, n_classes):  # This goes from 1:n_classes-1 -> class "0" is ignored
    # pred_inds = targets == cls
    # target_inds = target == cls
    intersection = preds.sum()  # Cast to long to prevent overflows
    union = preds.sum() + inputs.sum() - intersection
    if union == 0:
        iou = float('nan')  # If there is no ground truth, do not include in evaluation
    else:
        iou = float(intersection) / float(max(union, 1))
    return iou

def dice_coeff(inputs: Tensor, targets: Tensor, smooth=1):
#comment out if your model contains a sigmoid or equivalent activation layer
    inputs = torch.sigmoid(inputs)       
    
    #flatten label and prediction tensors
    inputs = inputs.view(-1)
    targets = targets.view(-1)
    
    intersection = (inputs * targets).sum()                            
    dice_value = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

    return dice_value



#PyTorch
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE

