# Alzheimer-MRI-CNN
Alzheimer's detection from MRI scans using a CNN built with Keras/TensorFlow. Classifies images into 4 stages: Mild, Moderate, Non-Demented, and Very Mild.

Overview

This project uses a Convolutional Neural Network (CNN) to classify brain MRI scans into four different stages of Alzheimer's disease: Non-Demented, Very Mild Demented, Mild Demented, and Moderate Demented. The model is implemented using Keras (with a TensorFlow backend) and trained on the Alzheimer's MRI Dataset.

Key Features

Data Preprocessing: Uses ImageDataGenerator to load, rescale, and augment images directly from directories.
Data Augmentation: Applies rotation, zooming, and horizontal flipping to the training data to improve model generalization.
Validation Split: Automatically splits the training data, using 20% for validation during training.
Model Design: A feedforward CNN with three Conv2D/MaxPooling2D blocks followed by Dense layers.
Multi-Class Classification: Uses a softmax activation function for 4-class classification and categorical_crossentropy as the loss function.
Model Saving: Saves the final trained model as Alzheimer_model.keras.

Project Steps

1. Setup and Requirements
    Installs tensorflow, keras, and matplotlib.
2. Data Processing
    Mounts Google Drive to access the dataset.
    Defines paths for train and test directories.
    Sets up ImageDataGenerator with a 20% validation split.
    Creates training, validation, and test generators using .flow_from_directory().
3. Model Training
    Defines a Sequential CNN model.
    Compiles the model for multi-class classification.
    Trains the model for 15 epochs.
4. Evaluation & Visualization
    Evaluates the model on the unseen test set.
    Plots "Model Accuracy" and "Model Loss" graphs.
    Generates and plots a Confusion Matrix.
    Prints a Classification Report (precision, recall, f1-score) for all 4 classes.

Setup Instructions

1.  Clone the Repository
    git clone [https://github.com/Hassansyed21/alzheimer-mri-cnn.git](https://github.com/Hassansyed21/alzheimer-mri-cnn.git)
    cd alzheimer-mri-cnn
2. Install Dependencies
    pip install -r requirements.txt
3. Run the Notebook
    Open alzheimer_mri_model.ipynb in Jupyter or Google Colab.
    Important: This project requires the pre-trained model file Alzheimer_model.keras. Download it from the link in the File Structure section and place it in your project folder.
    You must also upload the Alzheimer _MRI_Dataset to your environment.

File Structure
alzheimer-mri-cnn. 
|-- README.md. <-- This file

|-- alzheimer_mri_model.ipynb <-- The main project notebook

|-- Alzheimer_model.keras <-- (Model file is too large for GitHub)
                                 [Download from Google Drive](https://drive.google.com/file/d/1XMLnkaPANInessWGMyurNLTmHVndbiXC/view?usp=sharing)

|-- requirements.txt. <-- Python libraries

|-- .gitignore. <--Files to ignore

|-- LICENSE. <--MIT License

Results
Test Accuracy: ~60.82%.

Classification Report: The model shows varied performance across classes, with the best performance on the 'Non-Demented' and 'Very_Mild_Demented' categories.

Note: The accuracy and validation plots show that training accuracy (~77%) is higher than validation accuracy (~66%), which suggests the model is slightly overfitting

Note on Dataset
This model was trained on the "Alzheimer's MRI Dataset". Due to its large size, the dataset is not included in this repository.

The notebook is configured to load the data from a Google Drive path (`/content/drive/MyDrive/DATA SET/Alzheimer _MRI_Dataset `). To run this project, you must download the dataset separately and update this path in the notebook.

A popular version of this dataset can be found on Kaggle:

(https://www.kaggle.com/datasets/marcopinamonti/alzheimer-mri-4-classes-dataset)

License
This project is licensed under the MIT License see the LICENSE file for details
