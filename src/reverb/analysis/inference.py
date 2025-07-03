from collections import defaultdict

import torch
import tqdm
from ..training.pl_models import SupervisedSegmentationModule, create_segmentation_module
from ..training.dataloaders import RawDataDataset
from ..training.scaling import ClipZeroOneScaler

import segmentation_models_pytorch as smp

class ML_Model_Inference:

    labels = {0: 'background', 1: 'Resonance'}

    def __init__(self, path_to_checkpoint, device = 'cpu', scaler = None, model_kwargs = None,):
    # don't forget to initialize base class...

        self.device = device
        self.padding = (0, 4, 0, 17)
        if scaler is None:
            self.scaler = ClipZeroOneScaler(-250, -150)
        else:
            self.scaler = scaler

        self.model = self.load_my_model(path_to_checkpoint, model_kwargs)

    def load_my_model(self, path, model_kwargs = None):  
        if model_kwargs is None:
            num_classes = 2  # Assuming you have 2 classes (binary segmentation)

            model = smp.Unet(
                encoder_name="resnet18",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights=None,     # use `imagenet` pre-trained weights for encoder initialization
                in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=num_classes,                      # model output channels (number of classes in your dataset)
            )
        else:
            print(model_kwargs)
            model = create_segmentation_module(model_kwargs)
        
        segmentation_module = SupervisedSegmentationModule.load_from_checkpoint(
                                    path,
                                    model=model.model, 
                                    criterion="", 
                                    num_classes=2, 
                                    class_weights="",
                                    device=self.device,
                                    lr=0.0001,
                                    weight_decay=0.0001
                            ).to(self.device)
    
        return segmentation_module
    
    def predict(self, image):
        
        image = self.scaler(image)
        image = image.unsqueeze(0).to(self.device)
        mask = self.model(image)
        mask = mask.squeeze(0).cpu().detach().numpy()
        mask = mask[:, :-self.padding[3], self.padding[1]:]
        return mask

    def predict_batch(self, images):
        
        images = self.scaler(images)
        images = images.to(self.device)
        masks = self.model(images)
        masks = masks.cpu().detach().numpy()
        masks = masks[:, :, :-self.padding[3], self.padding[1]:]
        return masks
    
from torch.utils.data import Dataset
import re
import datetime

class SimpleNameImageDataset(Dataset):

    def __init__(self, h5_file_paths, padding):
        """
        """
        self.images = h5_file_paths
        self.image_names = list([str(path.name) for path in h5_file_paths])
        
        self.padding = padding

        # self.datetimes = []

        # pattern = r'(\d{4}\.\d{2}\.\d{2}-\d{6})'
        # for filename in h5_file_paths:
        #     match = re.search(pattern, str(filename))
        #     start_datetime = datetime.datetime.strptime(match.group(), "%Y.%m.%d-%H%M%S")
        #     self.datetimes.append(start_datetime)
    
    def _get_image(self, idx):
        img_path = self.images[idx]
        img = RawDataDataset.load_image(img_path, self.padding)
        
        return img

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img = self._get_image(idx)
        return self.image_names[idx], img


from pathlib import Path
from torch.utils.data import DataLoader
import joblib
import h5py
import numpy as np

def find_continuous_blocks(arr):
    padded_arr = np.pad(arr, pad_width=1, mode='constant')
    changes = np.diff(padded_arr)
    
    start_indices = np.where(changes == 1)[0]
    end_indices = np.where(changes == -1)[0]
    
    blocks = list(zip(start_indices, end_indices))
    
    return blocks

class MaskGenerator:

    def __init__(self, model, image_dataset, n_workers = 10, batch_size = 15):
        self.model = model

        self.n_workers = n_workers
        self.batch_size = batch_size
        self.dataloader = DataLoader(image_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers, collate_fn=self.custom_collate_datetimes)
        self.padding = image_dataset.padding
        self.num_files = len(image_dataset)

    @staticmethod
    def custom_collate_datetimes(batch):
        datetimes, images = zip(*batch)
        return datetimes, torch.vstack(images)

    @staticmethod
    def save_full_array(filename, array, array_name = 'mask'):
        #save using hdf5
        with h5py.File(filename, 'w') as f:
            f.create_dataset(array_name, data=array)
        

    def create_and_save_masks_to_folder(self, output_folder = None, bounding_box_output_dir = None):

        if output_folder is not None:
            output_folder = Path(output_folder)
            output_folder.mkdir(parents=True, exist_ok=True)

        if bounding_box_output_dir is not None:
            bounding_box_output_dir = Path(bounding_box_output_dir)
            bounding_box_output_dir.mkdir(parents=True, exist_ok=True)

        progress_bar = tqdm.tqdm("Predicted and saving resonances: ",total=self.num_files)
        with torch.no_grad():
            for files, images in self.dataloader:
                images = images.unsqueeze(1)[:, :, :416, :]
                masks = self.model.predict_batch(images)
                images = images.squeeze(1)
                predictions = np.argmax(masks, axis=1)
                resonant_energy_mask = predictions * images[:, :-self.padding[3], self.padding[1]:].detach().cpu().numpy()
                if output_folder is not None:
                    output_filenames = [output_folder / filename for filename in files]
                    joblib.Parallel(n_jobs=self.n_workers)(joblib.delayed(self.save_full_array)(filename, image) for filename, image in zip(output_filenames, resonant_energy_mask))
                
                if bounding_box_output_dir is not None:
                    output_filenames = [bounding_box_output_dir / (str(Path(filename).stem) + '.npy') for filename in files]
                    joblib.Parallel(n_jobs=self.n_workers)(joblib.delayed(self.gen_and_save_bounding_box)(filename, mask) for filename, mask in zip(output_filenames, predictions))

                progress_bar.update(len(files))
        progress_bar.close()

    @staticmethod
    def gen_and_save_bounding_box(filename, mask):
        bounding_boxes = MaskGenerator.bounding_box_generator(mask)
        np.save(filename, bounding_boxes)

    @staticmethod
    def bounding_box_generator(mask):
        start_end_pairs = []
        for row in range(mask.shape[0]):
            blocks = find_continuous_blocks(mask[row])
            for block in blocks:
                start_end_pairs.append((row, block[0], block[1]))
        if len(start_end_pairs) == 0:
            return np.array([])
        return np.vstack(start_end_pairs)
    
class WhaleFinder(MaskGenerator):

    def __init__(self, model, image_dataset, n_workers=10, batch_size=15):
        super().__init__(model, image_dataset, n_workers, batch_size)
    
    def find_and_save_whale_calls(self, output_folder, frequency_mask = None, threshold = 0):

        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        
        progress_bar = tqdm.tqdm("Predicted and saving resonances: ",total=self.num_files)
        with torch.no_grad():
            for files, images in self.dataloader:
                images = images.unsqueeze(1)[:, :, :416, :]
                masks = self.model.predict_batch(images)
                images = images.squeeze(1)
                predictions = np.argmax(masks, axis=1)
                predictions = predictions.astype('int64')
                if frequency_mask is not None:
                    predictions *= frequency_mask[np.newaxis, :, np.newaxis]
                num_whale_predictions_per_image = np.sum(predictions, axis=(1, 2))
                whale_masks = predictions[num_whale_predictions_per_image > threshold]

                whale_files = np.array(files)[num_whale_predictions_per_image > threshold]
                output_filenames = [output_folder / filename for filename in whale_files]
                if len(output_filenames) > 0:
                    joblib.Parallel(n_jobs=self.n_workers)(joblib.delayed(self.save_full_array)(filename, image) for filename, image in zip(output_filenames, whale_masks))
                progress_bar.update(len(files))
        progress_bar.close()
    def filter_significant_calls(self, output_folder, frequency_mask = None):

        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        
        for files, images in self.dataloader:
            images = images.unsqueeze(1)[:, :, :416, :]
            masks = self.model.predict_batch(images)
            images = images.squeeze(1)
            predictions = np.argmax(masks, axis=1)
            predictions = predictions.astype('int64')
            if frequency_mask is not None:
                predictions *= frequency_mask[np.newaxis, :, np.newaxis]
            num_whale_predictions_per_image = np.sum(predictions, axis=(1, 2))
            whale_masks = predictions[num_whale_predictions_per_image > 0]

            whale_files = np.array(files)[num_whale_predictions_per_image > 0]
            output_filenames = [output_folder / filename for filename in whale_files]
            if len(output_filenames) > 0:
                joblib.Parallel(n_jobs=self.n_workers)(joblib.delayed(self.save_full_array)(filename, image) for filename, image in zip(output_filenames, whale_masks))


class TemplateMatcher(MaskGenerator):

    def __init__(self, model, image_dataset, n_workers=10, batch_size=15):
        super().__init__(model, image_dataset, n_workers, batch_size)

    def compute_template_matches(self, template_matcher, model_hook_dict):
        results = []
        for files, images in self.dataloader:
            images = images.unsqueeze(1)
            _ = self.model.predict_batch(images)
            features = model_hook_dict['features']
            features = features.cpu().detach()
            scores = template_matcher.match_template(features)
            results.append((files, scores.detach().cpu()))
        
        return results