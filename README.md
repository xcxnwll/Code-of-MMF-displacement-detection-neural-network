# Code-of-MMF-displacement-detection-neural-network
Displacement Recognition Code

This folder contains two Python scripts for displacement recognition tasks. The two scripts are:

1. Recognitionnew.py: Divides the dataset into training, validation, and test sets, trains, and evaluates the model.
2. Displacementtest.py: Uses separate training and test datasets, trains, evaluates the model, and generates a confusion matrix.

Required Libraries
- torch
- torchvision
- matplotlib
- seaborn
- scikit-learn
- numpy

To install the dependencies, run:

    pip install torch torchvision matplotlib seaborn scikit-learn numpy

Dataset Structure

For Recognitionnew.py:  
The dataset should be in a single folder containing subfolders for each class:

    /input/
        /Displacement_1/
        /Displacement_2/
        ...

For Displacementtest.py:  
The dataset should be split into separate directories for training and validation:

    /train_test_data/
        /Displacement_1/
        /Displacement_2/
        ...
    /val_data/
        /Displacement_1/
        /Displacement_2/
        ...

File Descriptions

1. Recognitionnew.py

- Dataset Loading and Preprocessing:  
  Loads the dataset from a single folder structure. The speckles are resized, converted to grayscale, and normalized before being used for training.

- Data Splitting:  
  The dataset is split into three parts: training, validation, and test sets using random_split. The training set is 70%, validation is 10%, and the test set is 20%.

- CNN Model Definition:  
  A Convolutional Neural Network (CNN) with four convolutional layers and max-pooling operations is defined. Tanh activations and dropout are used to reduce overfitting.

- Model Training:  
  The model is trained for a specified number of epochs. During training, the loss and accuracy are tracked for both training and validation sets.

- Model Saving:  
  The model’s parameters are saved to a file (Net.pth), which allows reloading the model without retraining.

- Plotting Metrics:  
  The script generates and saves two plots:
  1. Loss plot: Shows training and validation loss over epochs.
  2. Accuracy plot: Shows training and validation accuracy over epochs.

- Test Set Evaluation:  
  The model is evaluated on the test set. The test loss and accuracy are computed and displayed.

- Confusion Matrix:  
  A confusion matrix is computed for the test set and visualized as a heatmap. The confusion matrix is saved as an image.

2. Displacementtest.py

- Dataset Loading:  
  Loads separate directories for the training and validation datasets. The speckles are resized, converted to grayscale, and normalized before being used for training.

- Data Splitting:  
  The training dataset (/train_test_data) is split into training and validation sets using random_split. The validation set is then used for evaluation during training, while the test set is loaded from the separate /val_data directory.

- CNN Model Definition:  
  Similar to Recognitionnew.py, a CNN with four convolutional layers, max-pooling, and dropout is defined.

- Model Training:  
  The model is trained for a specified number of epochs. During each epoch, training loss and accuracy are computed, and the model is evaluated on the validation set.

- Test Set Evaluation:  
  After training, the model is evaluated on the test set. The test loss and accuracy are computed and displayed.

- Confusion Matrix:  
  A confusion matrix is computed for the test set and visualized using a heatmap. The confusion matrix is saved as a PNG image.


