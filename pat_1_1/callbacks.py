# -*- coding: utf-8 -*-
"""
#################################################################################
    Paper ID   : 12076
    Title      : Dynamic Probabilistic Pruning: Training sparse networks based on stochastic and dynamic masking
#################################################################################
    Source Name    : callbacks.py
    Description    : This file containsthe callbacks to visualize logits (sample_vis), as well as validation 
                     loss and accuracy plots (training_vis).
                     - sample_vis: use this callback to visualize how logits and weights at the beginning 
                     and at the of the training.
                     - training_vis: use this callback to plot validation accuracy and loss at the end of
                     training.
#################################################################################
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

class sample_vis(tf.keras.callbacks.Callback):     
    def on_train_begin(self, epoch, logs=None):

        D1 = self.model.layers[1].get_weights()[2]
        W1 = self.model.layers[1].get_weights()[0]
        
        D2 = self.model.layers[2].get_weights()[2]
        W2 = self.model.layers[2].get_weights()[0]

        plt.figure(figsize=(8,20))
        plt.subplot(121)
        plt.imshow(D1,cmap='jet')
        plt.title('Logits layer 1')
        plt.subplot(122)
        plt.imshow(W1,vmin=-1,vmax=1,cmap='jet')
        plt.title('Weight layer 1')
        plt.show()
        
        plt.figure(figsize=(8,20))
        plt.subplot(121)
        plt.imshow(D2,cmap='jet')
        plt.title('Logits layer 2')
        plt.subplot(122)
        plt.imshow(W2,vmin=-1,vmax=1,cmap='jet')
        plt.title('Weight layer 2')
        plt.show()
        #### additional plot
        
        plt.figure(figsize=(10,10))
        plt.subplot(241)
        D1_1 = np.reshape(D1[0][:], (28,28))
        plt.imshow(D1_1,cmap='jet')
        plt.subplot(242)
        D1_2 = np.reshape(D1[50][:], (28,28))
        plt.imshow(D1_2,cmap='jet')
        plt.subplot(243)
        D1_3 = np.reshape(D1[100][:], (28,28))
        plt.imshow(D1_3,cmap='jet')
        plt.subplot(244)
        D1_4 = np.reshape(D1[150][:], (28,28))
        plt.imshow(D1_4,cmap='jet')
        plt.subplot(245)
        D1_5 = np.reshape(D1[200][:], (28,28))
        plt.imshow(D1_5,cmap='jet')
        plt.subplot(246)
        D1_6 = np.reshape(D1[250][:], (28,28))
        plt.imshow(D1_6,cmap='jet')
        plt.subplot(247)
        D1_7 = np.reshape(D1[299][:], (28,28))
        plt.imshow(D1_7,cmap='jet')
        plt.show()
        

        
        
    def on_train_end(self, epoch, logs=None):
        D1 = self.model.layers[1].get_weights()[2]
        W1 = self.model.layers[1].get_weights()[0]
        
        D2 = self.model.layers[2].get_weights()[2]
        W2 = self.model.layers[2].get_weights()[0]

        plt.figure(figsize=(8,20))
        plt.subplot(121)
        plt.imshow(D1,cmap='jet')
        plt.title('Logits layer 1')
        plt.subplot(122)
        plt.imshow(W1,vmin=-1,vmax=1,cmap='jet')
        plt.title('Weight layer 1')
        plt.show()
        
        plt.figure(figsize=(8,20))
        plt.subplot(121)
        plt.imshow(D2,cmap='jet')
        plt.title('Logits layer 2')
        plt.subplot(122)
        plt.imshow(W2,vmin=-1,vmax=1,cmap='jet')
        plt.title('Weight layer 2')
        plt.show()
        #### additional plot
        
        plt.figure(figsize=(10,10))
        plt.subplot(241)
        D1_1 = np.reshape(D1[0][:], (28,28))
        plt.imshow(D1_1,cmap='jet')
        plt.subplot(242)
        D1_2 = np.reshape(D1[50][:], (28,28))
        plt.imshow(D1_2,cmap='jet')
        plt.subplot(243)
        D1_3 = np.reshape(D1[100][:], (28,28))
        plt.imshow(D1_3,cmap='jet')
        plt.subplot(244)
        D1_4 = np.reshape(D1[150][:], (28,28))
        plt.imshow(D1_4,cmap='jet')
        plt.subplot(245)
        D1_5 = np.reshape(D1[200][:], (28,28))
        plt.imshow(D1_5,cmap='jet')
        plt.subplot(246)
        D1_6 = np.reshape(D1[250][:], (28,28))
        plt.imshow(D1_6,cmap='jet')
        plt.subplot(247)
        D1_7 = np.reshape(D1[299][:], (28,28))
        plt.imshow(D1_7,cmap='jet')
        plt.show()
        

        
    
class training_vis(tf.keras.callbacks.Callback):

    # This function is called when the training begins
    def on_train_begin(self, logs={}):
        # Initialize the lists for holding the logs, losses and accuracies
        self.losses = []
        self.acc = []
        self.val_losses = []
        self.val_acc = []
        self.logs = []

    # This function is called at the end of each epoch
    def on_epoch_end(self, epoch, logs={}):

        # Append the logs, losses and accuracies to the lists
        self.logs.append(logs)
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('categorical_accuracy'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_categorical_accuracy'))


    def on_train_end(self, epoch, logs={}):

        N = np.arange(0, len(self.losses))

         # Plot train loss, train acc, val loss and val acc against epochs passed
        fig, axs = plt.subplots(2, 1)
        axs[0].plot(self.losses, label = "train_loss")
        axs[0].plot(self.val_losses, label = "val_loss")
        axs[0].set_title("Training: Validation Loss and Accuracy".format(epoch))
        axs[0].set_xlabel("Epoch #")
        axs[0].set_ylabel("Loss")
        axs[0].legend()
        
        axs[1].plot(self.acc, label = "train_acc")
        axs[1].plot(self.val_acc, label = "val_acc")
        axs[1].set_xlabel("Epoch #")
        axs[1].set_ylabel("Accuracy")
        axs[1].legend()
        plt.show()
        



