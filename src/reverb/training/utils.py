import json
import numpy as np
import torch
from pathlib import Path
from torch import nn
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import glob
import os

from .scaling import ClipZeroOneScaler
from .dataloaders import (
    RawDataDataset, SyntheticDataDataset, UnsupervisedSemanticSegmentationDataset,
    SemiSupervisedDataloader
)
from .pl_models import SupervisedSegmentationModule
from .semi_supervised_pl import MeanTeacherSupervisionModel
from ..analysis.inference import ML_Model_Inference
from .pl_models import SupervisedSegmentationModule, create_segmentation_module


torch.set_float32_matmul_precision('medium')

# ============================
# Configuration & Utilities
# ============================

DEFAULT_MODEL_KWARGS = {
    "model_type": smp.Unet,
    "encoder_name": "resnet18",
    "encoder_weights": "imagenet",
    "in_channels": 1,
    "classes": 2,
}

DEFAULT_DATA_KWARGS = {
    "padding": (0, 4, 0, 17),
    "img_scaler": ClipZeroOneScaler(-250, -150),
    "feature_class": "resonance",
    # "mask_scaler": lambda mask: np.where(mask == 3, 1, 0),
    "unlabelled_dir": "../../../_labelling_/spectrogam_generation/unsupervised_files",
    "synthetic_noise_kwargs": {
        "noise_level": 0.04,
        "perlin_noise_scale": 0.03,
        "perlin_frequency": [10, 20],
        "pad": (0, 4, 0, 17)
    }
}

DEFAULT_TRAINING_KWARGS = {
    'lr': 2e-4,
    'weight_decay': 0.01,
    'batch_size': 10,
    'num_workers': 10,
    'max_epochs': 25,
    'class_weights': [0.5, 1.0],  # Example class weights
}


def load_annotations_json(root_dir):
    with open(f"{root_dir}/result.json", 'r') as f:
        result = json.load(f)
    return result


def get_split_indices(n, seed=0, train_ratio=0.8):
    idx = np.arange(n)
    np.random.seed(seed)
    np.random.shuffle(idx)
    split = int(train_ratio * n)
    return idx[:split], idx[split:]

# Mapping: 0 = null, 1 = resonance, 2 = whale
def remap_to_three_classes(mask):
    remapped = torch.zeros_like(mask)
    remapped[mask == 3] = 1  # resonance
    remapped[mask == 5] = 2  # whale
    return remapped

def one_hot_encoding_mask_three_classes(mask):
    mask = remap_to_three_classes(mask).long()
    one_hot_mask = torch.zeros((3,) + mask.shape, dtype=torch.long)
    one_hot_mask = one_hot_mask.scatter(0, mask.unsqueeze(0), 1)
    return one_hot_mask

blue_whale_call_bands = [[16,18]]
spec_frequencies = np.linspace(1.01,50.01, 399)
whale_frequencies_mask = np.zeros(399, dtype='int64')
for band in blue_whale_call_bands:
    start, end = band
    mask_indices = np.where((spec_frequencies >= start) & (spec_frequencies <= end))
    whale_frequencies_mask[mask_indices] = 1
whale_frequencies_mask = whale_frequencies_mask[::-1][:, np.newaxis]
padding = (0, 0, 0, 17)  # Padding for the spectrograms
pad_width = ((padding[2], padding[3]),   # (top, bottom) → (0, 17)
             (padding[0], padding[1]))   # (left, right) → (0, 0)

padded_whale_frequencies = np.pad(
    whale_frequencies_mask,
    pad_width=pad_width,
    mode='constant',
    constant_values=0
)
def whale_truncated_mask(mask):
    """
    Truncates the mask to only include the whale class (5).
    """
    whale_mask = np.where(mask == 5, 1, 0)
    # cut out regions outside 17 hz
    truncated_mask = whale_mask * padded_whale_frequencies
    
    return truncated_mask

variable_mask_functions = {
    "resonance": lambda mask: np.where(mask == 3, 1, 0),
    "whale": lambda mask: np.where(mask == 5, 1, 0),
    "multiclass": one_hot_encoding_mask_three_classes,
    "whale_truncated": whale_truncated_mask
}

def get_supervised_dataloaders(root_dir, data_kwargs, indices=None, batch_size=10, num_workers=10):
    result = load_annotations_json(root_dir)
    if indices is None:
        indices = get_split_indices(len(result['images']))
    mask_labels_processing_function = variable_mask_functions.get(data_kwargs['feature_class'], lambda x: x)

    dataset_args = dict(
        root_dir=root_dir,
        processing_function=data_kwargs['img_scaler'],
        mask_labels_processing_function=mask_labels_processing_function,
        pad=data_kwargs['padding']
    )

    train_dataset = RawDataDataset(indices=indices[0], **dataset_args)
    valid_dataset = RawDataDataset(indices=indices[1], **dataset_args)

    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    )


def get_semi_supervised_dataloaders(root_dir, data_kwargs, batch_size=10, num_workers=10):
    result = load_annotations_json(root_dir)
    labelled_idx = get_split_indices(len(result['images']))

    unlabelled_path = Path(data_kwargs['unlabelled_dir'])
    unlabelled_total = 5000
    unlabelled_idx = get_split_indices(unlabelled_total, train_ratio=0.9)

    mask_labels_processing_function = variable_mask_functions.get(data_kwargs['feature_class'], lambda x: x)
    train_labeled = RawDataDataset(root_dir, labelled_idx[0], data_kwargs['img_scaler'], mask_labels_processing_function, data_kwargs['padding'])
    val_labeled = RawDataDataset(root_dir, labelled_idx[1], data_kwargs['img_scaler'], mask_labels_processing_function, data_kwargs['padding'])

    train_unlabeled = UnsupervisedSemanticSegmentationDataset(unlabelled_path, unlabelled_idx[0], data_kwargs['img_scaler'], data_kwargs['padding'])
    val_unlabeled = UnsupervisedSemanticSegmentationDataset(unlabelled_path, unlabelled_idx[1], data_kwargs['img_scaler'], data_kwargs['padding'])

    return (
        SemiSupervisedDataloader(train_labeled, train_unlabeled, batch_size, 10 * batch_size, num_workers),
        SemiSupervisedDataloader(val_labeled, val_unlabeled, batch_size, 10 * batch_size, num_workers)
    )


def get_synthetic_dataloaders(root_dir, background_dir, data_kwargs, batch_size=10, num_workers=10):
    result = load_annotations_json(root_dir)
    indices = get_split_indices(len(result['images']))
    background_files = np.array(list(Path(background_dir).glob('*.h5')))
    mask_labels_processing_function = variable_mask_functions.get(data_kwargs['feature_class'], lambda x: x)
    synthetic_type = 'whale' if 'whale' in data_kwargs['feature_class'] else 'resonance'
    # Parameters for the image and calls
    if synthetic_type == 'whale':
        image_width = 60
        image_height = 399
        vertical_region = (int((1 - 0.34) * image_height) - 4, int((1 - 0.34) * image_height) + 4)
        feature_kwargs = dict(vertical_region=vertical_region, image_width=image_width, image_height=image_height, amplitudes =(10,40))
    else:
        feature_kwargs = {}
    train_dataset = SyntheticDataDataset(background_files,
                                         processing_function=data_kwargs['img_scaler'],
                                         mask_labels_processing_function=lambda x: x,
                                         raw_image_shape=(399, 60),
                                         padding=data_kwargs['padding'],
                                         noise_kwargs=data_kwargs['synthetic_noise_kwargs'],
                                         synthetic_type=synthetic_type,
                                         feature_kwargs=feature_kwargs)

    val_dataset = RawDataDataset(root_dir,
                                 indices=indices[1],
                                 processing_function=data_kwargs['img_scaler'],
                                 mask_labels_processing_function=mask_labels_processing_function,
                                 pad=data_kwargs['padding'])

    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    )


# ============================
# Training Function Factory
# ============================

def create_segmentation_module(model_kwargs, criterion = nn.CrossEntropyLoss(reduction='none'), class_weights = None, lr = 2.e-4, weight_decay=0.0, model_path = None,):
    if class_weights is None:
        class_weights = torch.Tensor(model_kwargs.get('class_weights', [1.0, 1.0])).to("cuda" if torch.cuda.is_available() else "cpu")


    model_type = model_kwargs.pop('model_type', smp.Unet)
    model = model_type(**model_kwargs)
    if model_path is None:
        return SupervisedSegmentationModule(model, criterion, model_kwargs['classes'], class_weights, lr, weight_decay)
    else:
        return SupervisedSegmentationModule.load_from_checkpoint(
            model_path, model=model, criterion=criterion,
            num_classes=model_kwargs['classes'], class_weights=class_weights,
            lr=lr, weight_decay=weight_decay)

def train(run_name, mode, root_dir="data", background_dir="data/low_res_backgrounds",
          model_kwargs=None, data_kwargs=None, training_kwargs=None, batch_size=10, num_workers=10, pretrain_path=None):

    model_kwargs = {**DEFAULT_MODEL_KWARGS, **(model_kwargs or {})}
    data_kwargs = {**DEFAULT_DATA_KWARGS, **(data_kwargs or {})}
    training_kwargs = {**DEFAULT_TRAINING_KWARGS, **(training_kwargs or {})}
    batch_size = training_kwargs.get('batch_size', batch_size)
    num_workers = training_kwargs.get('num_workers', num_workers)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    class_weights = torch.Tensor(training_kwargs.get('class_weights')).to(device)
    criterion = nn.CrossEntropyLoss(reduction='none')
    checkpoint_cb = ModelCheckpoint(
        dirpath=f'./checkpoints/{run_name}',
        filename='best_model',
        save_top_k=1,
        monitor='val_loss',
        mode='min'
    )
    callbacks = [checkpoint_cb]

    # Mode-specific logic
    if mode == "supervised":
        train_dl, val_dl = get_supervised_dataloaders(root_dir, data_kwargs, batch_size=batch_size, num_workers=num_workers)
        if pretrain_path is not None:
            pretrain_path = find_model_checkpoint(pretrain_path, './checkpoints', checkpoint_type="last")
            print(f"Using pre-trained model from: {pretrain_path}")
        model = create_segmentation_module(model_kwargs, criterion, class_weights, lr=training_kwargs['lr'], weight_decay=training_kwargs['weight_decay'], model_path=pretrain_path)

    elif mode == "semi_supervised":
        train_dl, val_dl = get_semi_supervised_dataloaders(root_dir, data_kwargs, batch_size, num_workers)
        model_type = model_kwargs.pop('model_type', smp.Unet)
        model = MeanTeacherSupervisionModel(
            model_type, model_kwargs, pad=data_kwargs['padding'], alpha=training_kwargs.get('alpha', 0.995),
            consistency_lambda=training_kwargs.get('consistency_lambda', 0.5), consistency_ramp_up=training_kwargs.get('consistency_ramp_up', 5),
            lr=training_kwargs['lr'], num_classes=model_kwargs['classes'], class_weights=class_weights,
            criterion=criterion, weight_decay=training_kwargs['weight_decay'], noise_kwargs=data_kwargs.get("synthetic_noise_kwargs", {})
        )

    elif mode == "synthetic_pretrain":
        train_dl, val_dl = get_synthetic_dataloaders(root_dir, background_dir, data_kwargs, batch_size, num_workers)
        model = create_segmentation_module(model_kwargs, criterion, class_weights, lr=training_kwargs['lr'], weight_decay=training_kwargs['weight_decay']) # 1e-3
        last_ckpt_cb = ModelCheckpoint(
            dirpath=f'./checkpoints/{run_name}',
            filename='last_model',
            save_last=True,  # <--- this is key
        )
        callbacks.append(last_ckpt_cb)
    else:
        raise ValueError(f"Unknown training mode: {mode}")

    # Logger & Trainer
    wandb_logger = WandbLogger(project='specgram_seg', name=run_name)
    trainer = pl.Trainer(
        max_epochs=training_kwargs['max_epochs'],
        accelerator=device,
        logger=wandb_logger,
        callbacks=callbacks
    )

    trainer.fit(model, train_dl, val_dl)
    wandb.finish()



### evaluation


def find_model_checkpoint(run_name, checkpoint_dir='./checkpoints', checkpoint_type='best'):
    if checkpoint_type not in ['best', 'last']:
        raise ValueError(f"Invalid checkpoint_type '{checkpoint_type}'. Use 'best' or 'last'.")

    filename = 'best_model.ckpt' if checkpoint_type == 'best' else 'last_model.ckpt'
    pattern = os.path.join(checkpoint_dir, run_name, filename)
    matches = glob.glob(pattern)

    if not matches:
        raise FileNotFoundError(f"No {checkpoint_type} checkpoint found for run: {run_name}")
    
    return matches[0]



def compute_results_over_eval_sets(run_name, eval_dataloaders, model_kwargs=None, training_kwargs=None, checkpoint_dir='./checkpoints', threshold=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    segmentation_module = load_best_model_from_run(run_name, model_kwargs, training_kwargs, checkpoint_dir)

    results = {}
    with torch.no_grad():
        segmentation_module.eval()
        for name, eval_dataloader in eval_dataloaders.items():
            for batch in eval_dataloader:
                images, masks = batch
                images, masks = images.to(device), masks.to(device)
                y_pred = segmentation_module(images)
                loss, loss_dict = segmentation_module.evaluate_losses(masks, y_pred, threshold)
                for k, v in loss_dict.items():
                    loss_dict[k] = v.detach().cpu().item()
                results[name] = loss_dict
                break  # Evaluate only first batch
    
    return results

def load_best_model_from_run(run_name, model_kwargs, training_kwargs, checkpoint_dir='./checkpoints'):

    # Locate the checkpoint path
    path_to_model = find_model_checkpoint(run_name, checkpoint_dir)

    # Merge default configs
    model_kwargs = {**DEFAULT_MODEL_KWARGS, **(model_kwargs or {})}
    training_kwargs = {**DEFAULT_TRAINING_KWARGS, **(training_kwargs or {})}

    segmentation_module = load_trained_model(path_to_model, model_kwargs, training_kwargs)
    return segmentation_module

def load_trained_model(path_to_model, model_kwargs, training_kwargs):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    num_classes = model_kwargs.get('classes', 2)
    class_weights = torch.Tensor(training_kwargs.get('class_weights', [1.0] * num_classes)).to(device)
    criterion = nn.CrossEntropyLoss(reduction='none')

    try:
        model = ML_Model_Inference(path_to_model, model_kwargs=model_kwargs,  device=device)
        model_core = model.model
    except Exception:
        model_type = model_kwargs.pop('model_type', smp.Unet)

        model_core = MeanTeacherSupervisionModel.load_from_checkpoint(
            path_to_model,
            backbone=model_type,
            backbone_config=model_kwargs,
            pad=model_kwargs.get('padding', 0),
            alpha=training_kwargs.get('alpha', 0.995),
            consistency_lambda=training_kwargs.get('consistency_lambda', 0.5),
            consistency_ramp_up=training_kwargs.get('consistency_ramp_up', 5),
            lr=training_kwargs.get('lr', 0.0002),
            num_classes=num_classes,
            class_weights=class_weights,
            criterion=criterion,
            weight_decay=training_kwargs.get('weight_decay', 0.0001),
            noise_kwargs=model_kwargs.get('synthetic_noise_kwargs', {})
        ).model

    segmentation_module = SupervisedSegmentationModule(
        model_core, criterion, num_classes, class_weights,
        lr=training_kwargs.get('lr', 0.0002),
        weight_decay=training_kwargs.get('weight_decay', 0.0001)
    )
    
    return segmentation_module

def save_evaluation_results(run_name, eval_results):
    import json
    with open(f'./checkpoints/{run_name}/eval_results.json', 'w') as f:
        json.dump(eval_results, f, indent=4)

def get_eval_dataloader(root_dir, data_kwargs, indices, batch_size=128, num_workers=10):
    # Only returns the "valid" split DataLoader
    _, val_loader = get_supervised_dataloaders(
        root_dir=root_dir,
        data_kwargs=data_kwargs,
        indices=(indices, indices),
        batch_size=batch_size,
        num_workers=num_workers
    )
    return val_loader


def get_eval_dataloaders(feature_class="resonance"):
    # Define shared parameters
    data_kwargs = {
        "img_scaler": ClipZeroOneScaler(-250, -150),
        "padding": (0, 4, 0, 17),
        "feature_class": feature_class
    }

    # Load validation data
    result = load_annotations_json("data")
    _, valid_indices = get_split_indices(len(result['images']))
    valid_dataloader = get_eval_dataloader("data", data_kwargs, indices=valid_indices)

    # Load rr and up34 evaluation data
    rr_result = load_annotations_json("data/rr_eval")
    rr_eval_dataloader = get_eval_dataloader("data/rr_eval", data_kwargs, indices=np.arange(len(rr_result['images'])))

    up34_result = load_annotations_json("data/up34_eval")
    up34_eval_dataloader = get_eval_dataloader("data/up34_eval", data_kwargs, indices=np.arange(len(up34_result['images'])))
    return {'valid': valid_dataloader,
            'rr_eval': rr_eval_dataloader,
            'up34_eval': up34_eval_dataloader}