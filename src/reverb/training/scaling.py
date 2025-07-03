
import torch
import torch.nn.functional as F

from pathlib import Path
import numpy as np
class ClipZeroOneScaler():

    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, data):
        data = torch.clamp(data, self.min_val, self.max_val)
        return (data - self.min_val) / (self.max_val - self.min_val)
    
    def inverse(self, data):
        return data * (self.max_val - self.min_val) + self.min_val
        
class MeanNoiseScaler():

    MEAN_RESONANCE_FILE_PATH = Path(__file__).parent / 'data/mean_resonance_scale.txt'

    def __init__(self, clip_min, clip_max, padding) -> None:
        self.mean_noise_scale = torch.Tensor(np.loadtxt(self.MEAN_RESONANCE_FILE_PATH))
        self.padded_mean_noise_scale = F.pad(self.mean_noise_scale, pad=padding[2:], mode='constant', value=self.mean_noise_scale[-1])
        self.clip_min = clip_min
        self.clip_max = clip_max
    def __call__(self, data, ):
        data = torch.clamp(data, self.clip_min, self.clip_max)
        try:
            data -= self.padded_mean_noise_scale.unsqueeze(-1)
        except RuntimeError:
            data -= self.mean_noise_scale.unsqueeze(-1)
        return data/10