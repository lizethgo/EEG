# -*- coding: utf-8 -*-
"""
#################################################################################
    Paper ID     : 12076
    Title        : Dynamic Probabilistic Pruning: Training sparse networks based on stochastic and dynamic masking
#################################################################################
        
    Source Name  : lenet_300_100_main.py
    Description  : Main file for MNIST using Lenet300-100. The network is pruned based on Dynamic Probabilistic
                   Pruning (DPP). 

#################################################################################
"""

from __future__ import print_function, absolute_import, division, unicode_literals
from __future__ import absolute_import
from __future__ import division


import tensorflow as tf
import h5py
import matplotlib.pyplot as plt

tf.compat.v1.disable_eager_execution()


import numpy as np
from numpy.random import seed
seed(1)


########################################################################################################
########################################################################################################
# Activate the following lines for GPU's usage

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
# Currently, memory growth needs to be the same across GPUs
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

tf.config.experimental.set_visible_devices(gpus[3],'GPU')

########################################################################################################
########################################################################################################
#%% Load data 




'''
****************************************************************
****************************************************************
file:feature_transaltor.py
Author: Lizeth Gonzalez Carabarin
Institution: Eindhoven University of Technology and EPFL
Date: 28-Oct-2020
Description: This files translates from *.mat format to numpy a
             array format 12 EEG features belonging to a single patient
             Arrays are 21600 size.
            
****************************************************************
****************************************************************
'''
import numpy as np
import scipy.io

p = '2'

### Translating mat format to numpy arrays
f_energy = scipy.io.loadmat('./ID{}/Feature_Energy.mat'.format(p))
f_energy_a = np.array(f_energy['Feature_energy'])

f_coastl = scipy.io.loadmat('./ID{}/Feature_coastline.mat'.format(p))
f_coastl_a = np.array(f_coastl['Feature_coastline'])

f_alpha = scipy.io.loadmat('./ID{}/Feature_frequency_alpha.mat'.format(p))
f_alpha_a = np.array(f_alpha['Feature_frequency_alpha'])

f_delta = scipy.io.loadmat('./ID{}/Feature_frequency_delta.mat'.format(p))
f_delta_a = np.array(f_delta['Feature_frequency_delta'])

f_beta = scipy.io.loadmat('./ID{}/Feature_frequency_beta.mat'.format(p))
f_beta_a = np.array(f_beta['Feature_frequency_beta'])

f_gamma_1 = scipy.io.loadmat('./ID{}/Feature_frequency_gamma1.mat'.format(p))
f_gamma_1_a = np.array(f_gamma_1['Feature_frequency_gamma1'])

f_gamma_2 = scipy.io.loadmat('./ID{}/Feature_frequency_gamma2.mat'.format(p))
f_gamma_2_a = np.array(f_gamma_2['Feature_frequency_gamma2'])

f_gamma_3 = scipy.io.loadmat('./ID{}/Feature_frequency_gamma3.mat'.format(p))
f_gamma_3_a = np.array(f_gamma_3['Feature_frequency_gamma3'])

f_gamma_4 = scipy.io.loadmat('./ID{}/Feature_frequency_gamma4.mat'.format(p))
f_gamma_4_a = np.array(f_gamma_4['Feature_frequency_gamma4'])

f_theta = scipy.io.loadmat('./ID{}/Feature_frequency_theta.mat'.format(p))
f_theta_a = np.array(f_theta['Feature_frequency_theta'])

f_n_energy = scipy.io.loadmat('./ID{}/Feature_nonlinEnergy.mat'.format(p))
f_n_energy_a = np.array(f_n_energy['Feature_nonlinEnergy'])

f_var = scipy.io.loadmat('./ID{}/Feature_variance.mat'.format(p))
f_var_a = np.array(f_var['Feature_variance'])

x_test = np.concatenate((#f_energy_a[:,0:5000],
                          #f_coastl_a[:,0:5000],
                          #f_alpha_a[:,0:5000],
                          f_beta_a[:,0:5000],
                          f_delta_a[:,0:5000],
                          #f_gamma_1_a[:,0:5000],
                          #f_gamma_2_a[:,0:5000],
                          #f_gamma_3_a[:,0:5000],
                          #f_gamma_4_a[:,0:5000],
                          f_theta_a[:,0:5000],
                          #f_n_energy_a[:,0:5000],
                          f_var_a[:,0:5000]), axis=0)

x_train = np.concatenate((#f_energy_a[:,5000:21600],
                          #f_coastl_a[:,5000:21600],
                          #f_alpha_a[:,5000:21600],
                          f_beta_a[:,5000:21600],
                          f_delta_a[:,5000:21600],
                          #f_gamma_1_a[:,5000:21600],
                          #f_gamma_2_a[:,5000:21600],
                          #f_gamma_3_a[:,5000:21600],
                          #f_gamma_4_a[:,5000:21600],
                          f_theta_a[:,5000:21600],
                          #f_n_energy_a[:,5000:21600],
                          f_var_a[:,5000:21600]), axis=0)




labels = scipy.io.loadmat('./ID{}/label_vector.mat'.format(p))
labels_a = np.array(labels['label_vector'])


x_train_seizure = x_train[:,11649:11738+1]

print("x_train_seizure", np.shape(x_train_seizure))
print("x_train", np.shape(x_train))
x_train = np.concatenate((x_train,
                          np.repeat(x_train_seizure,20,1)), axis=1)
print("x_train after oversampling", np.shape(x_train))
#print("x_train", x_train)        
x_train = np.transpose(x_train)
x_test = np.transpose(x_test)

print(np.shape(x_train))
print(np.shape(x_test))



y_test = labels_a[0,0:5000]
y_train = labels_a[0,5000:21600]  
y_train = np.concatenate((y_train, np.ones(20*90)))




#unique, counts = np.unique(labels_a, return_counts=True)
#print(dict(zip(unique, counts)));
### there are 1220 1's againts 20380 0's
#
### index_1 will store the values of indexes when label is 1
### index [1] stores the actual indexes of 1's
#
#
#
### First sizure (array length) --> 620 values (starting at 3959 - 4579)
#seizure_1 = np.where(index_1[1] < 10000)
#
### Second sizure (array length) --> 600 values (starting at 15096 - 15696)
#seizure_2 = np.where(index_1[1] > 10000)




from tensorflow.keras.utils import to_categorical

# normalize inputs from 0-255 to 0-1
#x_train =  x_train / np.max(x_train)
#x_test = x_test /np.max(x_test)


mean = np.mean(x_train,axis=(0,1))
std = np.std(x_train, axis=(0, 1))
x_train = (x_train-mean)/(std+1e-7)
x_test = (x_test-mean)/(std+1e-7)

print(np.shape(x_train))
print(np.shape(x_test))



# one hot encode outputs
#y_train = to_categorical(y_train_1)
#y_test = to_categorical(y_test_1)

print(np.shape(y_train))
print(np.shape(y_test))
print(np.shape(x_train))
print(np.shape(x_test))

num_classes = 2



########################################################################################################
########################################################################################################
##%% Define sparseConnect Model
from sparseconnect import sparseconnect_layer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

n_nodes_1 = 50 # nr of active output nodes (layer 1)
n_connect_1 = 3 #nr of active active connections per output node (layer 1)
n_nodes_2 = 100 # nr of active output nodes (layer 2)
n_connect_2 = 7 #nr of active active connections per output node (layer 1)
n_nodes_3 = 10  # nr of active output nodes (classification layer)
n_connect_3 = 10 #nr of active active connections per output node (classification layer)
n_epochs = 80
N_in = np.shape(x_train)[-1]
N_out = np.size(y_train,-1)

x = Input(shape=(N_in,))
x1 = sparseconnect_layer(n_nodes_1,n_connect_1,activation='relu',n_epochs=n_epochs, tempIncr=5)(x)
x2 = sparseconnect_layer(n_nodes_2,n_connect_2,activation='relu',n_epochs=n_epochs, tempIncr=5)(x1)
#x3 = sparseconnect_layer(n_nodes_2,n_connect_2,activation='relu',n_epochs=n_epochs, tempIncr=5)(x2)
#y = sparseconnect_layer(n_nodes_3,n_connect_3,activation='softmax',n_epochs=n_epochs, tempIncr=5)(x2)
#y = Dense(2,activation='softmax')(x2)
y = Dense(1,activation='sigmoid')(x2)
model = Model(inputs=x, outputs=y)

model.summary()
  

########################################################################################################
########################################################################################################
#%% Start training
import callbacks


optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
#optimizer = tf.keras.optimizers.SGD(lr=0.001, momentum=0.9)
### Optimization --> use lr=0.005 instead of 0.001 leads to faster convergene

callbacks = [callbacks.save_model(),
             callbacks.training_vis()]

model.compile(optimizer = optimizer,
             loss='binary_crossentropy',
              metrics=['binary_accuracy'])


history = model.fit(x=x_train, y=y_train, batch_size=16, epochs=n_epochs, verbose=1,
          callbacks = callbacks,
          validation_data=(x_test,y_test))
          

          
#file=h5py.File('p_acc_0.9908000230789185.h5py','r')
#weight = []
#for i in range(len(file.keys())):
#    weight.append(file['weight'+str(i)][:])
#    
##weight = np.sign(weight)
#model.set_weights(weight)
#file.close()


y_pred_1 = model.predict(x_test)
print(y_pred_1)
y_pred = np.round(y_pred_1)
print(y_pred_1)
#y_pred_1=np.sum(y_pred_1,axis=1)
print(np.shape(y_pred_1))
#print(y_pred_1)


from sklearn.metrics import confusion_matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

specificity = tn / (tn+fp)
sensitivity = tp / (tp + fn)
print(specificity)
print(sensitivity)
print(tn)
print(fp)
print(fn)
print(tp)

unique, counts = np.unique(labels_a, return_counts=True)
print(dict(zip(unique, counts)));
## there are 1220 1's againts 20380 0's

## index_1 will store the values of indexes when label is 1
## index [1] stores the actual indexes of 1's
index_1 = np.where(y_test == 1)
print(index_1)
seizure_1 = np.where(index_1[0] <5000 )
#print("length 1:", np.shape(seizure_1[0]))
print('starting point', index_1[0][seizure_1[0][0]])
print('ending point', index_1[0][seizure_1[0][-1]])
#print(index_1[1][seizure_1[0][-1]])

index_1 = np.where(y_pred == 1)
print(index_1[0])
### faeture selection
from tensorflow.keras import backend as K
feature_sel = K.eval(model.layers[1].zeros)
feature_sel = np.sum(feature_sel,0)
print('energy', feature_sel[0])
print('coastline', feature_sel[1])
print('alpha', feature_sel[2])
print('beta', feature_sel[3])
print('delta', feature_sel[4])
print('gamma1', feature_sel[5])
print('gamma2', feature_sel[6])
print('gamma3', feature_sel[7])
print('gamma4', feature_sel[8])
print('theta', feature_sel[9])
print('nonl_e', feature_sel[10])
print('var', feature_sel[11])

