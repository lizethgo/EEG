These files contains the experiment for the classification task of MNIST based on LeNet300-100
- lenet_300_100_main.py is the main file for model building and training 
- sparseconnect.py contains the sparse layers as well as the main algorithm for DPP
- temperatureUpdate.py contains a function for the temperature parameters used for the softamax relaxation function
- callbacks.py generates plots for validation loss and accuracy at the end of the training 

Python version: 3.7.4
TensorFlow version: 2.0

Instructions for running:
python lenet_300_100_main.py