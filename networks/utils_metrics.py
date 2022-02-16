import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
# SR : Segmentation Result
# GT : Ground Truth

def collect_metrics(targets: Tensor, predicted: Tensor, num_classes = 1, eps=1e-5):
    tp = torch.sum(predicted * targets).item() # TP
    fp = torch.sum(predicted * (1 - targets)).item()  # FP
    fn = torch.sum((1 - predicted) * targets).item()  # FN
    tn = torch.sum((1 - predicted) * (1 - targets)).item() # TN
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    specificity = (tn + eps) / (tn + fp + eps)
    f1 = 2*(precision*recall)/(precision+recall)
    dice_c = dice_coeff(targets, predicted)
    iou = jaccard_index(targets,predicted)
    # print(recall, precision, f1, specificity, dice_c.item(), iou.item() )

    return recall, precision, f1, specificity, dice_c.item(), iou.item() 
def dice_coeff(inputs: Tensor, targets: Tensor, smooth=1):
#comment out if your model contains a sigmoid or equivalent activation layer
    targets = torch.sigmoid(targets) 
    
    #flatten label and prediction tensors
    inputs = inputs.view(-1)
    targets = targets.view(-1)
    
    intersection = (inputs * targets).sum()                            
    dice_value = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

    return dice_value

def jaccard_index(inputs: Tensor, targets: Tensor, smooth=1):
    targets = torch.sigmoid(targets) 
    #flatten label and prediction tensors
    inputs = inputs.view(-1)
    targets = targets.view(-1)
    intersection = (inputs * targets).sum()                            
    total = (inputs + targets).sum()
    union = total - intersection
    return (intersection + smooth)/(union + smooth)


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