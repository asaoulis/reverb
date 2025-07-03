import torch

def compute_precision_recall(truths, preds):

    tp = torch.sum((preds == 1) & (truths == 1))
    fp = torch.sum((preds == 1) & (truths == 0))
    fn = torch.sum((preds == 0) & (truths == 1))

    precision = tp.float() / (tp + fp + 1e-8)
    recall = tp.float() / (tp + fn + 1e-8)

    return precision, recall

def compute_miou(truths, preds):

    intersection = torch.sum((preds == 1) & (truths == 1))
    union = torch.sum((preds == 1) | (truths == 1))
    miou = intersection.float() / (union + 1e-8)

    return miou

