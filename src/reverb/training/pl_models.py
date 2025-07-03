"""Pytorch Lightning modules for training segmentation models.

Currently includes the base and supervised modules. 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl

import torch.optim as optim

from . import metrics
from torcheval.metrics.functional import multiclass_precision, multiclass_recall

def mask_to_one_hot(mask, num_classes):
    # Create an empty tensor for one-hot encoding
    one_hot = torch.zeros(mask.size(0), num_classes, mask.size(1), mask.size(2)).to(mask.device)

    # Get the class labels from the mask
    class_labels = torch.unique(mask)

    # Set the corresponding pixels in the one-hot tensor to 1
    for label in class_labels:

        one_hot[:, int(label), :, :] = (mask == label).float()


    return one_hot

class SegmentationModule(pl.LightningModule):
    def __init__(self, criterion, num_classes, class_weights, lr, weight_decay, one_hot=False, class_list= []):
        super().__init__()
        
        self.criterion = criterion
        self.num_classes = num_classes
        self.class_labels = torch.arange(num_classes)
        self.class_weights = class_weights
        self.lr = lr
        self.weight_decay = weight_decay
        self.one_hot = one_hot
        if num_classes > 2:
            self.one_hot=True
        self.class_list = class_list

    def forward(self, x):
        return self.model(x)

    def evaluate_losses(self, masks, y_pred, threshold = None):
        if not self.one_hot:
            masks_one_hot = mask_to_one_hot(masks, self.num_classes)
        else:
            masks_one_hot = masks
        loss, loss_dict =  self._compute_losses(y_pred, masks_one_hot, threshold)
        return loss,loss_dict

    def _compute_losses(self, y_pred, masks_one_hot, threshold=None):

        loss_dict = {}
        loss = self.criterion(y_pred, masks_one_hot.float())

        per_class_losses = self.compute_per_class_losses(masks_one_hot, loss)
        for class_idx, class_name in enumerate(self.class_list):
            loss_dict[f'{class_name}_loss'] = per_class_losses[class_idx].item()

        per_class_losses *= self.class_weights
        batch_loss = per_class_losses.mean() / masks_one_hot.numel()

        loss_dict['loss'] = batch_loss

        # Calculate accuracy
        if threshold is None:
            preds = torch.argmax(y_pred, dim=1)
        else:
            probs = F.softmax(y_pred, dim=1)  # shape: (B, 2, H, W)
            class1_prob = probs[:, 1, :, :]  # shape: (B, H, W)
            preds = (class1_prob > threshold).long()
        truths = torch.argmax(masks_one_hot, dim=1)
        per_class_acc = []
        # per class accuracies / recall / precision
        for class_idx in range(masks_one_hot.shape[1]):  # Loop over classes
            # Calculate accuracy for each class
            class_mask = (truths == class_idx)
            correct_predictions = torch.sum((preds == truths)[class_mask])
            total_predictions = torch.sum(class_mask)
            class_accuracy = correct_predictions.float() / (total_predictions + 1e-8)  # Add epsilon to avoid division by zero
            per_class_acc.append(class_accuracy.item())

        for class_idx, class_name in enumerate(self.class_list):
            loss_dict[f'{class_name}_acc'] = per_class_acc[class_idx]

        acc = (preds == truths).float().mean()
        loss_dict['acc'] = acc

        # Calculate precision and recall
        # y_pred_flatten = y_pred.permute(0, 2, 3, 1).flatten(start_dim=0, end_dim=2)
        # truths_flatten = truths.flatten()
        # precision = multiclass_precision(y_pred_flatten, truths_flatten, num_classes=self.num_classes, average=None)
        # recall = multiclass_recall(y_pred_flatten, truths_flatten, num_classes=self.num_classes, average=None)

        # for class_idx, class_name in enumerate(self.class_list):
        #     loss_dict[f'{class_name}_precision'] = precision[class_idx].item()
        #     loss_dict[f'{class_name}_recall'] = recall[class_idx].item()
        precision, recall = metrics.compute_precision_recall(truths, preds)
        loss_dict['precision'] = precision
        loss_dict['recall'] =  recall

        # Calculate mean intersection over union (MIoU)
        miou = metrics.compute_miou(truths, preds)
        loss_dict['miou'] = miou

        return batch_loss * masks_one_hot.shape[0], loss_dict

    def compute_per_class_losses(self, masks_one_hot, loss):
        per_class_losses = torch.zeros(self.num_classes).to(self.class_weights.device)

        # Iterate over each class and select the per-class loss
        for class_idx in range(masks_one_hot.shape[1]):
            class_loss = loss[masks_one_hot[:, class_idx, :, :] == 1]
            if class_loss.numel() > 0:
                class_loss = class_loss.sum()
            else:
                class_loss = torch.zeros(1).to(self.class_weights.device)
            per_class_losses[class_idx] = class_loss
        return per_class_losses

    def _log_losses(self, loss_dict, prefix=''):
        for key, value in loss_dict.items():
            self._log_loss(f'{prefix}{key}', value)
    
    def _log_loss(self, name, value):
        self.log(name, value, on_epoch=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

class SupervisedSegmentationModule(SegmentationModule):

    def __init__(self, model, criterion, num_classes, class_weights, lr, weight_decay=0, one_hot=False, class_list=[]):
        super().__init__(criterion, num_classes, class_weights, lr, weight_decay, one_hot, class_list)
        self.model = model

    def training_step(self, batch, batch_idx):
        images, masks = batch
        y_pred = self.model(images)
        loss, loss_dict = self.evaluate_losses(masks, y_pred)

        self._log_losses(loss_dict)
        return loss

    def validation_step(self, batch, batch_idx):
        
        images, masks = batch
        y_pred = self.model(images)
        loss, loss_dict = self.evaluate_losses(masks, y_pred)

        self._log_losses(loss_dict, 'val_')

        return loss
    
import torch
from torch import nn

import pytorch_lightning as pl
import segmentation_models_pytorch as smp

def create_segmentation_module(model_kwargs, criterion = nn.CrossEntropyLoss(reduction='none'), class_weights = None, lr = 2.e-4, weight_decay=0.0):
    if class_weights is None:
        class_weights = torch.Tensor(model_kwargs.get('class_weights', [1.0, 1.0])).to("cuda" if torch.cuda.is_available() else "cpu")
    model_type = model_kwargs.pop('model_type', smp.Unet)
    model = model_type(**model_kwargs)
    return SupervisedSegmentationModule(model, criterion, model_kwargs['classes'], class_weights, lr, weight_decay)