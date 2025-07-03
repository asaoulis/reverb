"""Various methods to load in data for the different training schemes.

"""

from torch.utils.data import Dataset
import os
from PIL import Image
from pycocotools.coco import COCO
import torch.nn.functional as F

import torch
import numpy as np
import h5py


from abc import ABC, abstractclassmethod
from .synthetic_resonances import SyntheticResonances


def load_image_mask(coco, img_id):
  img = coco.loadImgs(img_id)[0]
  cat_ids = coco.getCatIds()
  anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
  anns = coco.loadAnns(anns_ids)
  
  anns_img_mask = np.zeros((img['height'],img['width']))     
  

  for ann in anns:

      x,y,w,h = ann['bbox']
      x1, y1 = x, y
      x2, y2 = x + w, y
      x3, y3 = x + w, y + h
      x4, y4 = x, y + h

      # Create a segmentation polygon
      segmentation = [[x1, y1, x2, y2, x3, y3, x4, y4]]

      # Assign the segmentation to the annotation
      ann['segmentation'] = segmentation
      ann['area'] = w*h
      anns_img_mask = np.maximum(anns_img_mask,coco.annToMask(ann)*(ann['category_id'] + 1))

  return anns_img_mask

from pathlib import Path


class SemanticSegmentationDataset(ABC, Dataset):
    """Image (semantic) segmentation dataset."""

    def __init__(self, root_dir, indices, processing_function, mask_labels_processing_function, pad=(0, 0, 0, 1)):
        """
        """
        self.root_dir = root_dir
        self.coco = COCO(root_dir + '/result.json')
        self.pad = pad

        self.image_handles = {}

        idx = 0
        for img_id in indices:
          img_handle = self.coco.imgs[img_id]
          
          img = self.coco.loadImgs(img_handle['id'])[0]
          anns_ids = self.coco.getAnnIds(imgIds=img['id'])
          if len(anns_ids) > 0:
            self.image_handles[idx] = img_handle
            idx +=1

        self.processing_function = processing_function
        self.mask_processing_function = mask_labels_processing_function
    
    @staticmethod
    def _pad(img, pad=(0, 0, 0, 1)):
        return F.pad(img, pad=pad, mode='constant', value=0)

    def __len__(self):
        return len(self.image_handles)

    def __getitem__(self, idx):

        img, mask = self._get_image_and_mask_arrays(idx)

        img = self.processing_function(img)
        mask = self.mask_processing_function(mask)

        return img, mask

    @abstractclassmethod
    def _get_image_and_mask_arrays(self, idx):
        pass


class RawDataDataset(SemanticSegmentationDataset):
    
    def _get_image_and_mask_arrays(self, idx):
        image_handle = self.image_handles[idx]
        img_path = self._get_img_path(idx)
        img = self.load_image(img_path, self.pad)

        mask = torch.Tensor(load_image_mask(self.coco, image_handle['id']))
        mask = self._pad(torch.fliplr(mask), pad=self.pad)
        
        return img, mask

    @staticmethod
    def load_image(img_path, pad=(0, 0, 0, 1)):
        array = np.flipud(np.array(h5py.File(img_path, 'r')['spectrogram_array'])).copy()
        img = torch.Tensor(array).unsqueeze(0)
        return RawDataDataset._pad(img, pad)

    def _get_img_path(self, idx):
        filename = self.image_handles[idx]['file_name']
        
        file_path = Path(self.root_dir + '/' + filename)
        #remove identifier - keep the string after the first - char
        filename_stem = "-".join(file_path.stem.split('-')[1:])
        raw_data_path = str((Path(self.root_dir + '/raw') / f"{filename_stem}.h5").resolve())
        return raw_data_path



class UnsupervisedSemanticSegmentationDataset(Dataset):

    def __init__(self, root_dir, indices, processing_function, pad):
        """
        """
        self.root_dir = root_dir
        self.images = np.array(list(Path(root_dir).glob('*.h5')))[indices]

        self.processing_function = processing_function
        self.pad = pad
    
    
    def _get_image(self, idx):
        img_path = self.images[idx]
        img = RawDataDataset.load_image(img_path, self.pad)
        
        return img

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img = self._get_image(idx)

        img = self.processing_function(img)

        return img


class ImageDataDataset(SemanticSegmentationDataset):
    
    def _get_image_and_mask_arrays(self, idx):

        raw_img = np.array(Image.open(self._get_img_path(idx)))
        img = torch.Tensor(raw_img[:, :, :3]).permute(2, 0, 1)
        # img = F.pad(img, pad=(0, 0, 0, 1), mode='constant', value=0)
        mask = torch.Tensor(load_image_mask(self.coco, idx))
        mask = torch.fliplr(mask)
        # mask = F.pad(torch.Tensor(load_image_mask(self.coco, idx)), pad=(0, 0, 0, 1), mode='constant', value=0)

        return img, mask

    def _get_img_path(self, idx):
        filename = self.image_handles[idx]['file_name']
        return os.path.join(self.root_dir, filename)

import itertools

class SemiSupervisedDataloader(torch.utils.data.IterableDataset):

    def __init__(self, supervised_dataset, unsupervised_dataset, batch_size, unsup_batch_size, num_workers = 1, shuffle = True):
        """
        """
        super().__init__()

        self.batch_size = batch_size
        self.unsup_batch_size = unsup_batch_size

        if num_workers > 1:
            num_workers = num_workers // 2
        else:
            num_workers = 0


        self.supervised_dataloader = torch.utils.data.DataLoader(supervised_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        self.unsupervised_dataloader = torch.utils.data.DataLoader(unsupervised_dataset, batch_size=unsup_batch_size, shuffle=shuffle, num_workers=num_workers)
    
    def __iter__(self):
        supervised_iter = iter(self.supervised_dataloader)
        # iterate forever
        unsupervised_iter = iter(itertools.cycle(self.unsupervised_dataloader))
        

        while True:
            try:
                image, mask = next(supervised_iter)
                unlabelled_image = next(unsupervised_iter)
                yield (image, mask), unlabelled_image
            except StopIteration:
                break

    def __len__(self):
        return len(self.supervised_dataloader)

from .semi_supervised_pl import TeacherNoiseInjector
from .synthetic_resonances import WhaleCallGenerator

class SyntheticDataDataset(Dataset):

    def __init__(self, background_noise_files,  processing_function, mask_labels_processing_function, raw_image_shape, synthetic_type='resonance', padding=(0, 0, 0, 1), feature_kwargs = {}, noise_kwargs = {}):
        if synthetic_type == 'resonance':
            self.feature_generator = SyntheticResonances(num_resonances_mean=10, amplitude=4, image_shape=raw_image_shape)
        elif synthetic_type == 'whale':
            self.feature_generator = WhaleCallGenerator(**feature_kwargs)
        self.noise_generator = TeacherNoiseInjector(0.05, 0.03, 10, pad=padding) if len(noise_kwargs) == 0 else TeacherNoiseInjector(**noise_kwargs)
        self.background_noise_files = background_noise_files
    
        self.processing_function = processing_function
        self.mask_processing_function = mask_labels_processing_function

        self.padding = padding
        
    def __getitem__(self, idx):
        background_image = RawDataDataset.load_image(self.background_noise_files[idx], pad=(0,0,0,0))
        resonance_image, masks = self.feature_generator.generate(background_image[0])
        resonance_image = resonance_image.to(dtype=torch.float32)
        masks = torch.tensor(masks.copy(), dtype=torch.float32)
        resonance_image = self.processing_function(resonance_image)
        masks = self.mask_processing_function(masks)

        resonance_image = F.pad(resonance_image, pad=self.padding, mode='constant', value=0).unsqueeze(0)
        noisy_resonant_image = self.noise_generator.add_noise(resonance_image)
        masks = F.pad(masks, pad=self.padding, mode='constant', value=0)

        return noisy_resonant_image, masks

    def __len__(self):
        return len(self.background_noise_files)
