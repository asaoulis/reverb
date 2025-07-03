"""Pytorch Lightning Semi-supervised segmentation modules

Using mean teacher approach to train.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from .pl_models import SegmentationModule

class TeacherNoiseInjector:

    def __init__(self, noise_level, perlin_noise_scale, perlin_frequency, pad):
        self.noise_level = noise_level
        self.perlin_noise_scale = perlin_noise_scale
        self.perlin_frequency = perlin_frequency
        self.pad = pad

    def replace_padded_with_zeros(self, img, pad):
        try:
            img[:, :, :, :pad[0]] = 1
            img[:, :, :, -pad[1]:] = 1
            img[:, :, :pad[2]] = 1
            img[:, :, -pad[3]:] = 1
        except:
            img[:, :, :pad[0]] = 1
            img[:, :, -pad[1]:] = 1
            img[:, :pad[2]] = 1
            img[:, -pad[3]:] = 1  
        
        return img 

    def generate_random_horizontal_mask_batch(self, images):
        masks = []
        batch_size, _, height, width  = images.shape
        for _ in range(batch_size):
            # Generate random starting position for the mask
            mask = self.random_horizontal_mask(torch.randint(100, 250, (1,)), height, width)
            masks.append(mask)
        
        return torch.stack(masks, dim=0).unsqueeze(1)

    def random_horizontal_mask(self, band_height, height, width):
        start_y = torch.randint(0, int(height - band_height), (1,))
            
            # Create a mask array and set ones within the band and zeros outside
        mask = torch.zeros((height, width))
        mask[start_y:start_y+band_height, :] = 1.0
        return mask
    
    def generate_random_vertical_mask_batch(self, images):
        masks = []
        batch_size, _, height, width  = images.shape
        for _ in range(batch_size):
            # Generate random starting position for the mask
            mask = self.random_vertical_mask(torch.randint(20, 50, (1,)), height, width)
            masks.append(mask)
        
        return torch.stack(masks, dim=0).unsqueeze(1)

    def random_vertical_mask(self, band_width, height, width):
        start_y = torch.randint(0, int(width - band_width), (1,))
            
            # Create a mask array and set ones within the band and zeros outside
        mask = torch.zeros((height, width))
        mask[:, start_y:start_y+band_width] = 1.0
        return mask

    def generate_perlin_noise(self, data, vertical=False):
        perlin_noise_batch = []
        batch_size, _, height, width  = data.shape

        for _ in range(batch_size):
            noise = self.perlin_noise(height, width, vertical=vertical)
            perlin_noise_batch.append(noise)
        return torch.stack(perlin_noise_batch, dim=0).unsqueeze(1)

    def perlin_noise(self, height, width, vertical=False):
        lin_x = torch.linspace(0, 5, height) + torch.rand(1) * 10
        lin_y = torch.linspace(0, 5, width) + torch.rand(1) * 10
        x, y = torch.meshgrid(lin_x, lin_y, indexing='ij')
        if vertical:
            noise = (self.perlin_noise_scale* (torch.randn(1) + 3)/5)* (torch.sin(self.perlin_frequency[1] * y))
        else:
            noise = (self.perlin_noise_scale* (torch.randn(1) + 3)/5)* (torch.sin(self.perlin_frequency[0] * x))
        return noise
    
    def __call__(self, data):
        noise = self.generate_perlin_noise(data) * self.generate_random_horizontal_mask_batch(data) 
        vertical_perlin = self.generate_perlin_noise(data, vertical=True) * self.generate_random_vertical_mask_batch(data) * self.generate_random_horizontal_mask_batch(data)
        data =  data + torch.randn_like(data).to(data.device) * (self.noise_level + vertical_perlin + noise).to(data.device)
        # return data
        return torch.clamp(self.replace_padded_with_zeros(data, self.pad), 0, 1)

    def add_noise(self, data):
        _, height, width = data.shape
        noise = self.perlin_noise(height, width) * self.random_horizontal_mask(torch.randint(100, 250, (1,)), height, width)
        vertical_perlin = self.perlin_noise(height, width, vertical=True) *\
                            self.random_vertical_mask(torch.randint(20, 50, (1,)), height, width) # *\
                                # self.random_horizontal_mask(torch.randint(50, 250, (1,)), height, width)
        data =  data + torch.randn_like(data).to(data.device) * (self.noise_level + noise + vertical_perlin).to(data.device)
        return torch.clamp(self.replace_padded_with_zeros(data + noise, self.pad), 0, 1)
class MeanTeacherSupervisionModel(SegmentationModule):
    def __init__(self, backbone, backbone_config, alpha, consistency_lambda, consistency_ramp_up, pad, noise_kwargs = {}, **kwargs):
        super().__init__(**kwargs)
        self.teacher = backbone(**backbone_config)

        self.teacher_noise_injector = TeacherNoiseInjector(0.05, 0.1, (10,20), pad) if len(noise_kwargs) == 0 else TeacherNoiseInjector(**noise_kwargs)
        for param in self.teacher.parameters():
            param.requires_grad = False
            
        self.model = backbone(**backbone_config)
        self.alpha = alpha  # EMA decay factor

        # Initialize teacher weights to be the same as the student weights
        self.teacher.load_state_dict(self.model.state_dict())
        self.ramp_up_epochs = consistency_ramp_up
        self.consistency_lambda = consistency_lambda

    def forward(self, x):
        # Forward pass through student model
        return self.model(x)

    def training_step(self, batch, batch_idx):

        loss_dict, total_loss = self.forward_step(batch)

        self._log_losses(loss_dict)

        return total_loss


    def validation_step(self, batch, batch_idx):

        loss_dict, total_loss = self.forward_step(batch)

        self._log_losses(loss_dict, prefix='val_')

        return total_loss
    

    def forward_step(self, batch):
        labeled_data, unlabeled_data = batch
        images, masks = labeled_data
        # Forward pass through student and teacher models with augmentation
        student_labelled_outputs = self.model(images)
        student_unlabelled_outputs = self.model(self.add_teacher_noise(unlabeled_data))
        
        supervised_loss, loss_dict = self.evaluate_losses(masks, student_labelled_outputs)
        teacher_outputs = self.teacher(unlabeled_data)


        # Compute the unsupervised loss with consistency regularization
        consistency_loss = F.mse_loss(student_unlabelled_outputs, teacher_outputs)
        current_epoch = self.trainer.current_epoch
        consistency_loss = self.consistency_lambda * consistency_loss * min(current_epoch / self.ramp_up_epochs, 1.0)# Ramp up factor
        # Total loss as a combination of supervised and unsupervised losses
        total_loss = (supervised_loss + consistency_loss)
        
        loss_dict['consistency_loss'] = consistency_loss
        loss_dict['total_loss'] = total_loss
        return loss_dict, total_loss
    
    def add_teacher_noise(self, data):
        return self.teacher_noise_injector(data)

    def on_after_backward(self):
        # Update the EMA weights of the teacher model
        with torch.no_grad():
            for teacher_param, student_param in zip(self.teacher.parameters(), self.model.parameters()):
                teacher_param.data.mul_(self.alpha).add_(student_param.data, alpha=1 - self.alpha)

