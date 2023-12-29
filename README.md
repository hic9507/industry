# Anomaly Detection in Industrial Environments
These codes are used to anomaly detection in industrial settings.
Anomaly Detection is performed using ResNet50 of TensorFlow environment.

## Only need to change the path of the dataset and graph and weights.

crop: Convert one image to 3x3 (300, 300) size images.

resize: Resize original image to (256, 256) size.

original: Use original images.

### Activate virtual environment.
conda activate your_environment

### Navigate to the code directory.
cd your/code/directory

## Enter a command and run these code.

### swir: semiconductor anomaly detection
#### swir_crop.py: 
To train 3x3 images.
#### swir_resize.py:
To train resized original images.
#### swir_original.py:
To train original images.

### window: windows(창호) anomaly detection
#### window_crop.py: 
To train 3x3 images.
#### window_resize.py:
To train resized original images.
#### window_original.py:
To train original images.
