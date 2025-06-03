README.md
markdown
Copy
Edit
# Tire Defect Detection Model Using YOLOv8

This project implements a tire defect detection system using the YOLOv8 object detection model. It is designed to automatically identify defects in tire images, helping to improve quality control in manufacturing and maintenance.

## Project Overview

Tire defects can lead to serious safety hazards and economic losses. This repository contains a deep learning model trained on a labeled dataset of tire images to classify tires as either **good** or **defective** using YOLOv8 architecture.

The model can assist automotive industries and maintenance teams by providing fast and accurate defect detection, minimizing manual inspection effort.

## Features

- Uses YOLOv8 for real-time object detection and classification.
- Dataset includes labeled images categorized as `good` and `defective`.
- Supports training, validation, and testing phases with proper dataset splits.
- Provides easy-to-use scripts for training and inference.
- Achieves high accuracy with optimized hyperparameters.

## Dataset

- The dataset consists of approximately 2000 images.
- Images are annotated with bounding boxes for defects.
- Split into training, validation, and test sets.
- Dataset prepared using Roboflow for consistent labeling and augmentation.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ranjeni7/Tire_defect_detection_model_using_yolov8.git
   cd Tire_defect_detection_model_using_yolov8
Create and activate a Python virtual environment (optional but recommended):

bash
Copy
Edit
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
The main dependencies include ultralytics (for YOLOv8), opencv-python, numpy, and other common ML libraries.

Usage
Training the model
Modify the configuration and dataset paths in the training script if needed. Then run:

bash
Copy
Edit
python train.py --data tire_dataset.yaml --epochs 50 --batch 16 --img 640
Running inference
To test the trained model on new images or videos:

bash
Copy
Edit
python detect.py --weights runs/train/exp/weights/best.pt --source path_to_test_images/
Results
The trained YOLOv8 model achieves high precision and recall for defect detection.

Confusion matrices, loss curves, and sample inference images are provided in the /results folder.

Contributing
Contributions and suggestions are welcome! Feel free to open issues or pull requests.

License
This project is licensed under the MIT License.

Contact
Author: Ranjeni C

GitHub: ranjeni7

Email: cranjeni1@gmail.com
