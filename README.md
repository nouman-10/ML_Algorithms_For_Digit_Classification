# ML_Algorithms_For_Digit_Classification
Comparison of various algorithms for Digit Classification

### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)

## Installation <a name="installation"></a>

The libraries required for the successful execution of this code are mentioned in requirements.txt. In order to install all the libraries:
`pip install -r requirements.txt`

## Project Motivation<a name="motivation"></a>


In this project, the problem of image classification is tackled, specifically digit classification. The data contains 200 images for each digit (0-9) with 100 images for training and 100 for testing. 

Various Machine Learning and Deep Learning algorithms are compared such as Logistic regression, Support Vector Machines, Multi-layer Perceptron, and Convolutional Neural Network. Data augmentation is also explored here.
## File Descriptions <a name="files"></a>

- `helpers.py` contains various helper functions to read, pre-process, and plot the data. Furthermore, functions for training and testing various models are also included.

- `main.py` uses the functions from `helpers.py` to train and fit the model passed through the command line. There are two command line arguments, one for the model to be trained and second whether to apply data augmentation or not. For instance, to run the CNN model with data augmentation:

```python main.py -m cnn -d```

The different models are:
- lr: logistic regression
- svm: support vector machine
- mlp: multi-layer perceptron
- cnn: convolutional neural network


