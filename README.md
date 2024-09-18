# Face Mask Detection using CNN and Transfer Learning

This project focuses on detecting whether a person is wearing a face mask using Convolutional Neural Networks (CNN) and Transfer Learning techniques. The model is trained on images of people with masks, without masks, and with partial masks. Various CNN architectures like basic custom CNN, VGG16, and ResNet50 are used.

# Project Structure
Face_Mask_Detection.ipynb: The Jupyter notebook containing the full pipeline for data processing, model building, training, and evaluation.

# Requirements
Before running the code, ensure you have the following packages installed:

TensorFlow
Keras
scikit-learn
matplotlib
numpy
pandas
PIL (Python Imaging Library)
You can install the required dependencies using:

Copy code
```bash
pip install tensorflow keras scikit-learn matplotlib numpy pandas pillow
```

# Dataset
The dataset for this project contains three classes:

partial_mask: People wearing masks incorrectly.
with_mask: People wearing masks correctly.
without_mask: People without masks.
The dataset is automatically downloaded using the following command in the notebook:

python
Copy code
```bash
!wget -qq https://cdn.iisc.talentsprint.com/CDS/MiniProjects/MP2_FaceMask_Dataset.zip
!unzip -qq MP2_FaceMask_Dataset.zip
```

Download and extract the dataset.
Preprocess the data using ImageDataGenerator.
Build and train the models:
Custom CNN with 2 convolutional layers.
Enhanced CNN with multiple convolutional layers.
Transfer Learning models using VGG16 and ResNet50 pre-trained architectures.
Model Evaluation:

After training, the model is evaluated on validation data and the training/validation loss is plotted to visualize the performance.
Model Architectures
1. Custom CNN (Model 1)
Two convolutional layers followed by max-pooling and dense layers.
Trained for 5 epochs with binary_crossentropy loss.
2. Advanced CNN (Model 2)
A deeper architecture with four convolutional layers.
Trained for 5 epochs with better accuracy and loss compared to the basic model.
3. Transfer Learning (VGG16 and ResNet50)
Pre-trained VGG16 and ResNet50 architectures are used with custom dense layers for mask classification.
These models achieved high accuracy with fine-tuning and transfer learning.
Results
The custom CNN model achieves good initial performance with basic accuracy.
The transfer learning models (VGG16, ResNet50) outperform the custom CNN, achieving higher accuracy on the validation set.
Plots
The training and validation loss are plotted after each model training to show the model's learning over time.

Notes
Training time may vary depending on your hardware setup. Using GPUs (with TensorFlow-GPU) is recommended for faster training.



