


import gc
import joblib
import numpy as np
from tqdm import tqdm

import contextlib
from tqdm import tqdm

import gc

import torch
from pathlib import Path
import numpy as np

import torch
from reverb.training.dataloaders import RawDataDataset
from reverb.analysis.inference import ML_Model_Inference, MaskGenerator, SimpleNameImageDataset

import h5py
def load_resonant_energy_array(file):
    with h5py.File(file, 'r') as f:
        return np.array(f['mask'])

def count_number_of_predicted_resonances(resonant_energy_array):
    return np.count_nonzero(resonant_energy_array)
    

class MeanResonanceScale:

    def __init__(self, mean_width, image_folder, resonant_energy_folder):
        self.mean_width = mean_width
        self.image_folder = image_folder
        self.resonant_energy_folder = resonant_energy_folder
    
    def get_resonance_free_pixels(self, resonant_energy_array):
        return resonant_energy_array == 0
    
    @staticmethod
    def compute_per_freq_energy_sums_and_counts(image, noise_pixels):
        non_zero_counts = noise_pixels.sum(axis=1)
        row_sums = (image * noise_pixels).sum(dim=1)
        return row_sums, non_zero_counts


class ResonanceEnergyEstimator:

    def __init__(self, mean_width, image_folder, resonant_energy_folder, image_height):
        self.mean_width = mean_width
        self.image_folder = image_folder
        self.resonant_energy_folder = resonant_energy_folder
        self.energy_scale = np.linspace(1.01, 50.01, image_height)[::-1]
    

    def blur_resonances_from_image(self, file):
        image =  RawDataDataset.load_image(self.image_folder / file, pad=(0,0,0,0)).squeeze()
        resonant_energy_array = load_resonant_energy_array(self.resonant_energy_folder  / file)
        background_pixels = resonant_energy_array == 0
        # get indices of resonant pixels
        energy_row_sums, counts = MeanResonanceScale.compute_per_freq_energy_sums_and_counts(image, background_pixels)
        
        average_freq_energy = energy_row_sums / counts
        mean_resonance_scale = np.convolve(average_freq_energy, np.ones(self.mean_width), 'same') / self.mean_width
        expanded_mean = np.repeat(mean_resonance_scale[:, np.newaxis], image.shape[1], axis=1)
        
        resonant_energy = (image)*~background_pixels
        resonant_energy = resonant_energy[self.mean_width//2 + 1:-(self.mean_width//2+1)]
        
        return expanded_mean, resonant_energy, ~background_pixels

    def get_total_resonant_energy(self, file):
        scaled_energy = self.get_sum(file)
        sum = scaled_energy.sum().numpy()
        del scaled_energy
        return  sum

    def get_sum(self, file):
        mean, energy, mask = self.blur_resonances_from_image(file)
        mask = mask[self.mean_width//2 + 1:-(self.mean_width//2+1)]
        scaled_energy = np.power(10, 1/10 *energy) - np.power(10, 1/10 * mean[self.mean_width//2 + 1:-(self.mean_width//2+1)])

        scaled_energy *= self.energy_scale[self.mean_width//2 + 1:-(self.mean_width//2+1), np.newaxis]
        scaled_energy = scaled_energy * mask
        return scaled_energy
@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    def tqdm_print_progress(self):
        if self.n_completed_tasks > tqdm_object.n:
            n_completed = self.n_completed_tasks - tqdm_object.n
            tqdm_object.update(n=n_completed)

    original_print_progress = joblib.parallel.Parallel.print_progress
    joblib.parallel.Parallel.print_progress = tqdm_print_progress

    try:
        yield tqdm_object
    finally:
        joblib.parallel.Parallel.print_progress = original_print_progress
        tqdm_object.close()


from pathlib import Path
from collections import defaultdict
resonant_energy_folder =  Path("../../../_outputs_/_spec_data_/_spec_data_28_36_/spectrogram_data/")
resonant_energy_folder = Path("results/UP37_46_predictions/resonant_energy_arrays")
# Dictionary to store files by station
files_by_station = defaultdict(list)
resonant_energy_array_files = list(Path(resonant_energy_folder).glob("*.h5"))
# Populate the dictionary
for path in resonant_energy_array_files:
    filename = path.name
    station = filename.split('_')[0]  # Extract station (e.g., UP42)
    files_by_station[station].append(path.name)

# Sort the lists of paths for each station alphanumerically
for station in files_by_station:
    files_by_station[station].sort(key=lambda x: x)
    print(station, len(files_by_station[station]))
raw_data_dir = Path("/data/UPFLOW/projects/iReverb/_outputs_/_spec_data_/_spec_data_37_46/spectrogram_data")
h5_files = list(raw_data_dir.glob("*.h5"))
resonance_energy_estimator = ResonanceEnergyEstimator(mean_width=5, image_folder=Path(raw_data_dir), resonant_energy_folder=Path(resonant_energy_folder), image_height=399)


station_energies = {}
num_jobs = 20
batch_size = 10000  # Process in batches of 2000 files

for station, files in files_by_station.items():
    batched_results = []  # Store results for this station
    
    for batch_start in range(0, len(files), batch_size):
        batch_files = files[batch_start:batch_start + batch_size]

        if num_jobs != 1:
            with tqdm_joblib(tqdm(desc=f"Processing {station}", total=len(batch_files))) as progress_bar:
                with joblib.parallel_backend('loky', n_jobs=num_jobs):
                    batch_res = joblib.Parallel()(
                        joblib.delayed(resonance_energy_estimator.get_total_resonant_energy)(file)
                        for file in batch_files
                    )
        else:
            batch_res = [
                resonance_energy_estimator.get_total_resonant_energy(file) 
                for file in tqdm(batch_files, total=len(batch_files))
            ]
        
        # Convert results to numpy (reduces memory overhead)
        batch_res = np.array(batch_res, dtype=np.float32)  

        # Append batch results and periodically release memory
        batched_results.extend(batch_res.tolist())  

        # **Manual Memory Cleanup**
        del batch_res  # Remove batch-level variable
        gc.collect()   # Force garbage collection

    # Store all batches for this station
    station_energies[station] = batched_results

    # Additional memory cleanup
    del batched_results
    gc.collect()

# Save the results
joblib.dump(station_energies, "station_energies.pkl")