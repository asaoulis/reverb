"""
Some basic code to generate synthetic resonances for pre-training. Brownian motion idea is to emulate
the smoothly varying nature of the resonances.
"""
import numpy as np

class BrownianMotion:
    delta_t = 0.05
    lower_bound = 0.0
    upper_bound = 1.0

    def reflect_boundary(self, value, lower_bound, upper_bound):
        if value < lower_bound:
            return 2 * lower_bound - value
        elif value > upper_bound:
            return 2 * upper_bound - value
        return value

    def __call__(self, scale, num_points):

        delta_x = np.random.normal(0, scale=np.sqrt(self.delta_t), size=num_points)
        brownian_motion = np.cumsum(delta_x) + scale

        # Apply reflecting boundary condition
        bounded_brownian_motion = [self.reflect_boundary(x, self.lower_bound, self.upper_bound) for x in brownian_motion]
        if np.random.random() > 0.5:
            bounded_brownian_motion = np.flip(bounded_brownian_motion)
        return np.array(bounded_brownian_motion)

class SyntheticResonances:
    CHANCE_FULL_WIDTH = 0.3
    def __init__(self, num_resonances_mean, amplitude, image_shape):
        self.num_resonances_mean = num_resonances_mean
        self.amplitude = amplitude
        self.brownian_motion = BrownianMotion()

        self.image_shape = image_shape
        self.height = image_shape[0]
        first_weights_len = self.height//10 + self.height // 3
        self.weights  = np.concatenate([ np.ones(self.height - first_weights_len), np.ones(self.height // 3) * 8,np.zeros(self.height//10)])
        self.weights /= np.sum(self.weights) 
    
    def generate(self, image):
        resonances = np.zeros_like(image)
        masks = np.zeros_like(image)
        num_resonances = max(7, np.random.poisson(self.num_resonances_mean))
        
        resonance_indices = np.random.choice(self.height, num_resonances, p=self.weights, replace=False)
        
        amplitude_multiplier = 1* np.random.uniform(1, 1.5) * (1 + num_resonances/8)
        for index in resonance_indices:
            if np.random.random() > 0.2:
                resonances[index], masks[index] = self._create_single_resonance(amplitude_multiplier)
            else:
                resonance_height = np.maximum(np.random.poisson(2.5), 2)
                results = [self._create_single_resonance(3*amplitude_multiplier) for _ in range(resonance_height)]
                resonances[index:index+resonance_height] = np.vstack([outputs[0] for outputs in results])
                masks[index:index+resonance_height] = np.vstack([outputs[1] for outputs in results])

        return image + resonances, masks

    def _create_single_resonance(self, image_resonance_multiplier):

        rand = np.random.random()
        image_width = self.image_shape[1]
        if rand < self.CHANCE_FULL_WIDTH:
            start_idx = 0
            width = self.image_shape[1]
        else:
            min_width = 10
            start_idx = np.random.randint(0, image_width - min_width)
            width = np.random.randint(min_width, image_width - start_idx)

        resonance = np.zeros((image_width))
        mask = np.zeros((image_width), dtype=np.uint8)
        resonance[start_idx:start_idx+width] =  np.abs(image_resonance_multiplier * self.brownian_motion(self.amplitude , width))[:width] + self.amplitude/2
        mask[start_idx:start_idx+width] = 1
        return resonance, mask
    
import numpy as np
import random

class WhaleCallGenerator:
    def __init__(self, image_width=60, image_height=399, 
                        amplitudes=(1,2),
                         vertical_region=(20, 40), 
                         call_count_range=(8), 
                         spacing_range=(4, 10), 
                         margin=5):

        self.image_width = image_width
        self.amplitudes = amplitudes
        self.image_height = image_height
        self.vertical_region = vertical_region
        self.call_count_range = call_count_range
        self.spacing_range = spacing_range
        self.margin = margin


    def generate_teardrop_patch(self, width=4, height=6, tail=7, distortion_factor=0.2, noise_amplitude=0.3, skew_factor=0.5):
        """
        Generate a 6x10 pixel patch with a teardrop shape, optionally distorted and with added noise.

        Parameters:
        - width (int): Width of the patch in pixels.
        - height (int): Height of the patch in pixels.
        - distortion_factor (float): Factor to distort the teardrop shape.
        - noise_amplitude (float): Amplitude of random noise to add.
        - skew_factor (float): Factor to skew the shape upwards.

        Returns:
        - np.ndarray: A 2D array representing the teardrop patch.
        """
        # Initialize a blank patch
        patch = np.zeros((height + tail, width))

        # Define the teardrop center
        center_x = width / 2
        center_y = (height) / 2

        # Create the teardrop shape
        for y in range(height + tail):
            for x in range(width):
                x_pos = x + 0.5 
                y_pos = y + 0.5
                # Calculate distances and distortion, applying skew to vertical distance
                vertical_distance = ((y_pos - center_y) / height) - skew_factor * ((height - y_pos) / height)
                horizontal_distance = (x_pos - center_x) / width

                # Teardrop shape equation with distortion
                shape_value = 1 - (horizontal_distance ** 2 + vertical_distance ** 2 + distortion_factor * vertical_distance)
                vertical_tail =  (0.4 + np.abs(1 - vertical_distance/6 )) * np.abs(0.5 - (horizontal_distance)**2) if y_pos -1 > center_y else 0
                vertical_tail = np.clip(vertical_tail, 0, 1)
                if shape_value > 0:
                    patch[y, x] = shape_value
                if vertical_tail > 0:
                    patch[y, x] += vertical_tail

                
        # Add random noise
        noise = noise_amplitude * np.random.rand(height + tail, width)
        patch = (patch + noise)/np.max(patch + noise)

        return patch

    def generate(self, background):
        """
        Generate an image simulating blue whale calls.

        Parameters:
        - image_width (int): Width of the image in pixels.
        - image_height (int): Height of the image in pixels.
        - vertical_region (tuple): Range (min, max) of vertical pixels where calls appear.
        - call_count_range (tuple): Range of the number of calls.
        - spacing_range (tuple): Range of pixels between calls.
        - margin (int): Empty pixel band before and after the calls.

        Returns:
        - np.ndarray: Generated image array.
        """
        # Initialize a blank image
        
        image = np.zeros((self.image_height, self.image_width))
        mask = np.zeros((self.image_height, self.image_width))

        if random.random() < 0.8:
            return background, mask

        # Randomize the number of calls and their starting horizontal position
        call_count = np.random.poisson(self.call_count_range)
        curr_x = random.randint(0, self.margin)
        y_center = random.randint(*self.vertical_region)

        width = np.random.randint(2, 4)
        noise_amplitude = np.random.uniform(0.3, 0.7)
        tail = np.random.randint(2, 6)
        height=np.random.randint(3, 5)
        amplitude = np.random.uniform(*self.amplitudes)
        # Generate each call
        for i in range(call_count):
            # Randomize the horizontal position and spacing
            spacing = random.randint(*self.spacing_range)
            x_pos = curr_x + spacing
            curr_x = x_pos
            
            # Ensure the call doesn't exceed the image boundary
            if x_pos >= self.image_width - 3:
                break
            elif random.random() < 0.2:
                continue
            elif random.random() < 0.02:
                break

            # Randomize vertical position within the specified region
            y_pos = y_center + np.random.poisson(0.5) - 1
            
            patch = np.random.uniform(0.8, 1.2) * amplitude * self.generate_teardrop_patch(height=height, width=width, tail=tail, distortion_factor=0.4, noise_amplitude=noise_amplitude, skew_factor=0.7)
            patch_height, patch_width = patch.shape
            start_y = y_pos - patch_height // 2
            image[start_y:start_y + patch_height,  x_pos:x_pos + patch_width] += patch
            mask[start_y:start_y + patch_height,  x_pos:x_pos + patch_width] = 1
        if random.random() < 0.5:
            image = np.flip(image, axis=1)
            mask = np.flip(mask, axis=1)

        return background + image, mask