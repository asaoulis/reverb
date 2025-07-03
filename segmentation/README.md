# Segmentation Analysis and Model Training

## Dataset Generation

In `data_generation/`, there is some code for generation of the spectrogram images. These are required for manual annotations performed in label-studio, but can also be used directly for the ML model training.

Also included in this folder is a `.ini` file that was used to generate the spectrogram data from `UP05` using `OBSToolbox`. The key parameters chosen at this step is :

* Frequency range of `[1, 50]` Hz
* Spectrogram images have dimension `(400,60)` (freq, time),  covering 15 minutes of data, and with the frequency scaled linearly such that each pixel covers 0.05 Hz. 

## Model Training

`model_training/SemanticSegmentationTests.ipynb` contains some utilities for loading in COCO format segmentation images and masks. 

It then defines a segmentation ML model, trains it, and has some code to inspect its outputs at inference time. At the moment, it by default throws away all labels except for resonance / no resonance and performs binary classification on a pixel-wise level.
