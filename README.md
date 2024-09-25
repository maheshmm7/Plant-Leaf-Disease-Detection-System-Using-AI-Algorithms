# Plant Disease Detection System

Welcome to the Plant Disease Detection System! 🌿🔍

This project utilizes advanced image processing techniques through TensorFlow and Streamlit to identify plant diseases efficiently. By uploading an image of a plant, the system analyzes it to detect any signs of diseases.

## Table of Contents
- [Features](#features)
- [Data Processing](#data-processing)
- [Model Building](#model-building)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [About the Dataset](#about-the-dataset)
- [Installation](#installation)

## Features
- **Image Upload:** Users can upload images of plants for analysis.
- **Disease Recognition:** The model predicts and classifies the type of plant disease present.
- **User-Friendly Interface:** Built using Streamlit for a seamless experience.
- **Fast Processing:** Quick results with minimal waiting time.

## Data Processing

### Training and Validation Image Preprocessing
The dataset is organized into separate directories for training and validation. Images are resized to 128x128 pixels and processed using TensorFlow's `image_dataset_from_directory` method.

```python
training_dataset = tf.keras.utils.image_dataset_from_directory('path/to/train', ...)
validation_dataset = tf.keras.utils.image_dataset_from_directory('path/to/val', ...)
```

## Model Building
The model is a Convolutional Neural Network (CNN) constructed using TensorFlow Keras. It consists of multiple convolutional layers, max pooling layers, dropout layers, and a final dense layer with softmax activation for classification.

```python
model.add(Conv2D(...))
model.add(MaxPool2D(...))
model.add(Dense(...))
```

## Usage
1. Install the required libraries:
   ```bash
   pip install tensorflow matplotlib pandas seaborn streamlit
   ```

2. To run the application:
   ```bash
   streamlit run app.py
   ```

3. Open your browser and go to `http://localhost:8501` to access the application.

## How It Works
1. **Upload Image:** Upload an image of a plant with suspected diseases.
2. **Analysis:** The system processes the image using advanced algorithms.
3. **Results:** View the prediction results, indicating the type of disease detected.

## About the Dataset
This dataset consists of approximately 26,000 RGB images of healthy and diseased crop leaves, categorized into 11 different classes. The training set contains 20,815 images, the validation set includes 5,203 images, and there are 14 test images created for prediction purposes.

### Dataset Structure
- train: 20,815 images
- validation: 5,203 images
- test: 14 images

### Dataset Source
You can download the dataset from Kaggle: [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)

## Installation
Make sure to have Python installed and use pip to install the necessary packages as mentioned above. The dataset must be structured correctly in the specified directories for training and validation.
