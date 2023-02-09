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
import os
import h5py
from tensorflow.keras import backend as K

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
        self.acc.append(logs.get('binary_accuracy'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_binary_accuracy'))


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
        

class save_model(tf.keras.callbacks.Callback):



    # This function is called at the end of each epoch
    def on_epoch_end(self, epoch, logs={}):
        
        #self.val_acc.append(logs.get('val_categorical_accuracy'))

        # Append the logs, losses and accuracies to the lists
        print(logs.get('val_binary_accuracy'))

        if logs.get('val_binary_accuracy') > 0.6:
            
            mask = []
            j = 0
            k = 0
                    
            mask.append(K.eval(self.model.layers[1].zeros))
            mask.append(K.eval(self.model.layers[2].zeros))
            #mask.append(K.eval(self.model.layers[3].zeros))
            

            #np.savetxt("mask.csv", mask, delimiter=",")

            
            
            file = h5py.File('p_acc_{}.h5py'.format(logs.get('val_binary_accuracy')),'w')
            weight = self.model.get_weights()

            
            
             #for i in range(len(weight)):
                # #print(weight)
                # if i == 0 or i == 3 or i == 6:
                    # file.create_dataset('weight'+str(j),data=weight[i]*mask[k]) 
                    # j+=1
                    # k+=1
                # else :
                    # file.create_dataset('weight'+str(j),data=weight[i])    
                    # j+=1
            for i in range(len(weight)):
                file.create_dataset('weight'+str(i),data=weight[i])
            file.close()
            
            
            
            
#            file = h5py.File('lebet5_bin_MASK_acc{}.h5py'.format(logs.get('val_categorical_accuracy')),'w')
#            for i in range(len(mask)):
#                file.create_dataset('mask'+str(i),data=mask[i])    
#            file.close()






