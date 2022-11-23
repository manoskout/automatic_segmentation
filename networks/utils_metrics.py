import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from medpy import metric
import math
def assert_shape(test, reference):

    assert test.shape == reference.shape, "Shape mismatch: {} and {}".format(
        test.shape, reference.shape)


class ConfusionMatrix:

    def __init__(self, test=None, reference=None):

        self.tp = None
        self.fp = None
        self.tn = None
        self.fn = None
        self.size = None
        self.reference_empty = None
        self.reference_full = None
        self.test_empty = None
        self.test_full = None
        self.set_reference(reference)
        self.set_test(test)

    def set_test(self, test):

        self.test = test
        self.reset()

    def set_reference(self, reference):

        self.reference = reference
        self.reset()

    def reset(self):

        self.tp = None
        self.fp = None
        self.tn = None
        self.fn = None
        self.size = None
        self.test_empty = None
        self.test_full = None
        self.reference_empty = None
        self.reference_full = None

    def compute(self):

        if self.test is None or self.reference is None:
            raise ValueError("'test' and 'reference' must both be set to compute confusion matrix.")

        assert_shape(self.test, self.reference)

        self.tp = int(((self.test != 0) * (self.reference != 0)).sum())
        self.fp = int(((self.test != 0) * (self.reference == 0)).sum())
        self.tn = int(((self.test == 0) * (self.reference == 0)).sum())
        self.fn = int(((self.test == 0) * (self.reference != 0)).sum())
        self.size = int(np.prod(self.reference.shape, dtype=np.int64))
        self.test_empty = not np.any(self.test)
        self.test_full = np.all(self.test)
        self.reference_empty = not np.any(self.reference)
        self.reference_full = np.all(self.reference)

    def get_matrix(self):

        for entry in (self.tp, self.fp, self.tn, self.fn):
            if entry is None:
                self.compute()
                break

        return self.tp, self.fp, self.tn, self.fn

    def get_size(self):

        if self.size is None:
            self.compute()
        return self.size

    def get_existence(self):

        for case in (self.test_empty, self.test_full, self.reference_empty, self.reference_full):
            if case is None:
                self.compute()
                break

        return self.test_empty, self.test_full, self.reference_empty, self.reference_full

def collect_metrics(ground_truth: Tensor, predicted: Tensor, classes: dict = None, eps=1e-5):
    """The metrics collector function
    In binary returns a single float value for each metric.
    In multiclass return a list containing the values for each class that segmented
    """
    # del classes[0] # Delete the background
    # print(f"METRICS : {classes}")

    # In our test 0 : background, 1: rectum, 2: vessie, 3: femoral_l, 4: femoral_r TODO-> get rid of the hardcoded method
    # print(f"Predicted before torch.where : {torch.unique(predicted)}")
    predicted = torch.where(predicted>0.1,1,0).cpu().detach().numpy()
    # print(np.unique(predicted))
    
    # print(f"Predicted after torch.where : {np.unique(predicted)}")
    
    if not classes:
        ground_truth = torch.where(ground_truth>0,1,0).cpu().detach()
        precision_v = precision(predicted,ground_truth)
        recall_v = recall(predicted,ground_truth)
        specificity_v = specificity(predicted,ground_truth)
        sensitivity_v = sensitivity(predicted,ground_truth)
        hausdorff_distance_v = hausdorff_distance(predicted,ground_truth)
        hausdorff_distance_95_v =hausdorff_distance_95(predicted,ground_truth)
        avg_surface_distance_v = avg_surface_distance(predicted,ground_truth) 
        dice_c = dice(predicted,ground_truth)
        iou = jaccard(predicted,ground_truth)

        return recall_v, precision_v, specificity_v,sensitivity_v, dice_c, iou ,hausdorff_distance_v, hausdorff_distance_95_v, avg_surface_distance_v
    else:
        # print(f"Unique in ground truth: {torch.unique(ground_truth.squeeze())}")
        # print(f"Unique in predicted: {predicted.shape}")

        ground_truth = ground_truth.cpu().detach()

        precision_v = []
        recall_v = []
        specificity_v = []
        sensitivity_v = []
        hausdorff_distance_v = []
        hausdorff_distance_95_v = []
        dice_c = []
        iou = []
        avg_surface_distance_v = []
        for item, id in classes.items():   

            class_truth = np.where(ground_truth==id, 1, 0)[:,0,:,:]
            # print(f"For item : {item}, pred shape {predicted[:,id,:,:].shape}, gtr shape: {class_truth.shape}")
            # import matplotlib.pyplot as plt
            # plt.subplot(1,2,1)
            # plt.title("Predicted")

            # plt.imshow(predicted[:,id,:,:].squeeze())
            # plt.subplot(1,2,2)
            # plt.title("Ground Truth")
            
            # plt.imshow(class_truth.squeeze())

            # plt.show()
            # print(f"{id}, organ: {item}")
            # print(f"Unique in ground truth {np.unique(class_truth)} .... shape : {class_truth.shape}")
            # print(f"Unique in predicted {np.unique(predicted[:,id,:,:])} .... shape : {predicted[:,id,:,:].shape}")
            
            # Deal with the issue of the imbalanced dataset
            # if not math.isnan(dice(predicted[:,id,:,:],class_truth)) and dice(predicted[:,id,:,:],class_truth) != 0.:
            precision_v.append(precision(predicted[:,id,:,:],class_truth))
            recall_v.append(recall(predicted[:,id,:,:],class_truth))
            specificity_v.append(specificity(predicted[:,id,:,:],class_truth))
            sensitivity_v.append(sensitivity(predicted[:,id,:,:],class_truth))
            hausdorff_distance_v.append(hausdorff_distance(predicted[:,id,:,:],class_truth))
            hausdorff_distance_95_v.append(hausdorff_distance_95(predicted[:,id,:,:],class_truth))
            # print(dice(predicted[:,id,:,:],class_truth))
            dice_c.append(dice(predicted[:,id,:,:],class_truth))
            iou.append(jaccard(predicted[:,id,:,:],class_truth))
            avg_surface_distance_v.append(avg_surface_distance(predicted[:,id,:,:],class_truth))
            
        # print(precision_v, recall_v, specificity_v, sensitivity_v, hausdorff_distance_v, hausdorff_distance_95_v, dice_c, iou)
        return (
            np.array(recall_v), np.array(precision_v), np.array(specificity_v), 
            np.array(sensitivity_v),  np.array(dice_c), np.array(iou),
            np.array(hausdorff_distance_v), np.array(hausdorff_distance_95_v), np.array(avg_surface_distance_v)
        )
        



def hausdorff_distance(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, voxel_spacing=1.5, connectivity=1, **kwargs):

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if test_empty or test_full or reference_empty or reference_full:
        if nan_for_nonexisting:
            # https://github.com/MIC-DKFZ/nnUNet/issues/380
            return float('NaN')
        else:
            return 0

    test, reference = confusion_matrix.test, confusion_matrix.reference

    return metric.hd(test, reference, voxel_spacing, connectivity)

def avg_surface_distance(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, voxel_spacing=1.5, connectivity=1, **kwargs):

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if test_empty or test_full or reference_empty or reference_full:
        if nan_for_nonexisting:
            # https://github.com/MIC-DKFZ/nnUNet/issues/380
            return float('NaN')
        else:
            return 0

    test, reference = confusion_matrix.test, confusion_matrix.reference

    return metric.asd(test, reference, voxel_spacing, connectivity)


def hausdorff_distance_95(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, voxel_spacing=1.5, connectivity=1, **kwargs):

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if test_empty or test_full or reference_empty or reference_full:
        if nan_for_nonexisting:
            # https://github.com/MIC-DKFZ/nnUNet/issues/380
            return float('NaN')
        else:
            return 0

    test, reference = confusion_matrix.test, confusion_matrix.reference

    return metric.hd95(test, reference, voxel_spacing, connectivity)


def dice(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """2TP / (2TP + FP + FN)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    # print(f"Dice : TP : {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if test_empty and reference_empty:
        if nan_for_nonexisting:
            return float('NaN')
        else:
            return 0.
    dc = float(2. * tp / (2 * tp + fp + fn))
    return dc if dc!=0. else np.nan


def jaccard(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """TP / (TP + FP + FN)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    # print(f"IOU : TP : {tp}, FP: {fp}, TN: {tn}, FN: {fn}")

    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if test_empty and reference_empty:
        if nan_for_nonexisting:
            return float('NaN')
        else:
            return 0.
    jc = float(tp / (tp + fp + fn))
    return jc if jc!=0. else np.nan


def precision(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """TP / (TP + FP)"""
    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if test_empty:
        if nan_for_nonexisting:
            return float('NaN')
        else:
            return 0.

    return float(tp / (tp + fp))


def sensitivity(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """TP / (TP + FN)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()
    # https://github.com/MIC-DKFZ/nnUNet/issues/380 error about NAN value
    if reference_empty:
        if nan_for_nonexisting:
            return float('NaN')
        else:
            return 0.
    try:    
        sensitiv = float(tp / (tp + fn))
        return float(tp / (tp + fn))
    except ZeroDivisionError:
        return 0


def recall(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """TP / (TP + FN)"""

    return sensitivity(test, reference, confusion_matrix, nan_for_nonexisting, **kwargs)


def specificity(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """TN / (TN + FP)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if reference_full:
        if nan_for_nonexisting:
            return float('NaN')
        else:
            return 0.

    return float(tn / (tn + fp))


def accuracy(test=None, reference=None, confusion_matrix=None, **kwargs):
    """(TP + TN) / (TP + FP + FN + TN)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()

    return float((tp + tn) / (tp + fp + tn + fn))


class AverageMeter():
    """Computes and stored thhe average value of our metrics"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.loss = 0
        self.iou = 0
        self.hd = 0
        self.hd95 = 0
        self.dice = 0
        self.specificity = 0
        self.sensitivity = 0
        self.recall = 0
        self.precision = 0 
        self.avg_surface_distance = 0

        self.sum_loss = 0
        self.sum_iou = 0
        self.sum_hd = 0
        self.sum_hd95 = 0
        self.sum_dice = 0
        self.sum_specificity = 0
        self.sum_sensitivity = 0
        self.sum_recall = 0
        self.sum_precision = 0
        self.sum_asd =0 

        self.avg_loss = 0
        self.avg_iou = 0
        self.avg_hd = 0
        self.avg_hd95 = 0
        self.avg_dice = 0
        self.avg_specificity = 0
        self.avg_sensitivity = 0
        self.avg_recall = 0
        self.avg_precision = 0
        self.avg_asd = 0


        self.count = 0
        
    def update(self,loss: torch.Tensor,true_mask: torch.Tensor, pred_mask: torch.Tensor , n: int =1, classes: dict = None):
        """Collects all the metrics and calculate the average
        There are two different type of metrics [muliclass, binary],
        we setted the binary as predifined
        """
        # del classes[0]

        recall, precision, specificity, sensitivity, dice, iou , hd, hd95, as_distance =  collect_metrics(true_mask,pred_mask, classes)
        # print(dice)
        self.count += n

        self.loss = loss
        self.sum_loss += loss * n
        self.avg_loss = self.sum_loss / self.count


        self.iou = iou
        self.hd = hd
        self.hd95 = hd95
        self.dice = dice
        self.specificity = specificity
        self.sensitivity = sensitivity
        self.recall = recall
        self.precision = precision
        self.avg_surface_distance = as_distance

        

        # self.sum_iou += iou * n
        # self.sum_hd += hd * n
        # self.sum_hd95 += hd95 * n
        # self.sum_dice += dice * n
        # self.sum_specificity += specificity * n
        # self.sum_sensitivity += sensitivity * n
        # self.sum_recall += recall * n
        # self.sum_precision += precision * n

        
        # self.avg_iou = self.sum_iou / self.count
        # self.avg_hd = self.sum_hd / self.count
        # self.avg_hd95 = self.sum_hd95 / self.count
        # self.avg_dice = self.sum_dice / self.count
        # self.avg_specificity = self.sum_specificity / self.count
        # self.avg_sensitivity = self.sum_sensitivity / self.count
        # self.avg_recall = self.sum_recall / self.count
        # self.avg_precision = self.sum_precision /self.count


        if classes:
            # Get the mean from all the classes
            self.all_iou = np.nanmean(self.iou)
            self.all_hd = np.nanmean(self.hd)
            self.all_hd95 = np.nanmean(self.hd95)
            self.all_dice = np.nanmean(self.dice)
            self.all_specificity = np.nanmean(self.specificity)
            self.all_sensitivity = np.nanmean(self.sensitivity)
            self.all_recall = np.nanmean(self.recall)
            self.all_precision = np.nanmean(self.precision)
            self.all_asd = np.nanmean(self.avg_surface_distance)

        else:
            # TODO -> make it more readable 
            self.all_iou = self.avg_iou
            self.all_hd = self.avg_hd
            self.all_hd95 = self.avg_hd95
            self.all_dice = self.avg_dice
            self.all_specificity = self.avg_specificity
            self.all_sensitivity = self.avg_sensitivity
            self.all_recall = self.avg_recall
            self.all_precision = self.avg_precision
            self.all_asd = self.avg_avg_asd

        #print(f"n IOU: {self.iou}, of all classes: {self.all_iou}, Images :{self.count}, Sum: {self.sum_iou}, Avg: {self.avg_iou}")
        


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss