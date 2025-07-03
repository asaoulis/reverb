
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_single_tag_keys

import numpy as np
import segmentation_models_pytorch as smp

import torch
from pathlib import Path
from reverb.training.pl_models import SupervisedSegmentationModule
from reverb.training.dataloaders import RawDataDataset
from reverb.training.scaling import ClipZeroOneScaler

SEGMENTATION_LIB_PATH = Path(__file__).absolute().parent.parent.parent.parent / 'reverb'

PATH_TO_CHECKPOINT = SEGMENTATION_LIB_PATH / 'segmentation' / 'auto_labelling' / 'resnet18_pre_0.5' / 'best_model-val_loss=5557.47.ckpt'
RAW_DATA_PATH = SEGMENTATION_LIB_PATH / 'segmentation' / 'model_training' / 'data'/'raw'

def find_continuous_blocks(arr):
    padded_arr = np.pad(arr, pad_width=1, mode='constant')
    changes = np.diff(padded_arr)
    
    start_indices = np.where(changes == 1)[0]
    end_indices = np.where(changes == -1)[0]
    
    blocks = list(zip(start_indices, end_indices))
    
    
    return blocks

class MyModel(LabelStudioMLBase):

    labels = {0: 'background', 1: 'Resonance'}

    def __init__(self, **kwargs):
    # don't forget to initialize base class...

        super(MyModel, self).__init__(**kwargs)
        self.device = 'cpu'
        self.padding = (0, 4, 0, 17)

        self.scaler = ClipZeroOneScaler(-250, -150)

        self.model = self.load_my_model(PATH_TO_CHECKPOINT)

        self.from_name, self.to_name, self.value, self.labels_in_config = get_single_tag_keys(
            self.parsed_label_config, 'RectangleLabels', 'Image')

    def load_my_model(self, path):  

        num_classes = 2  # Assuming you have 2 classes (binary segmentation)

        model = smp.Unet(
            encoder_name="resnet18",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=None,     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=num_classes,                      # model output channels (number of classes in your dataset)
        )
        
        segmentation_module = SupervisedSegmentationModule.load_from_checkpoint(
                                    path,
                                    model=model, 
                                    criterion="", 
                                    num_classes=num_classes, 
                                    class_weights="",
                                    device=self.device,
                                    lr=0.0001,
                                    weight_decay=0.0001
                            ).to(self.device)
    
        return segmentation_module
    
    def predict(self, tasks, **kwargs):
        expected_location_path = Path(tasks[0]['data'][self.value])
        unique_filename_stem = str(expected_location_path.stem)
        filename = "-".join(unique_filename_stem.split('-')[1:]) + '.h5'
        
        image_path = RAW_DATA_PATH / filename

        raw_image = RawDataDataset.load_image(image_path, self.padding)
        image = self.scaler(raw_image)

        image = torch.Tensor(image[np.newaxis, ...])
        mask = self.model(image.to(self.device))[0, :,  :-self.padding[3], self.padding[1]:]
        
        predicted_label_idx = torch.argmax(mask, dim=0)
        predicted_label_idx = torch.fliplr(predicted_label_idx)

        predictions = []
        results = []
        img_width = 60
        img_height = 399

        for r in range(img_height):
            row_data = predicted_label_idx[r]
            blocks = find_continuous_blocks(row_data)
            for block in blocks:
                x, xmax = block
                width = xmax - x
                if width > img_width//4:
                    result = {
                        "original_width": img_width,
                        "original_height": img_height,
                        'from_name': self.from_name,
                        'to_name': self.to_name,
                        'type': 'rectanglelabels',
                        'value': 
                                {
                                'rectanglelabels': ['Resonance'],
                                'x': x / img_width * 100,
                                'y': r / img_height * 100,
                                'width': width / img_width * 100,
                                'height': 1 / img_height * 100
                                }
                    }
                    results.append(result)

        predictions.append({'result': results})
        return predictions