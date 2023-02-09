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



p = '4_1'

### Translating mat format to numpy arrays
f_energy_1 = scipy.io.loadmat('./ID{}/Feature_Energy.mat'.format(p))
f_energy_a_1 = np.array(f_energy_1['Feature_energy'])

f_coastl_1 = scipy.io.loadmat('./ID{}/Feature_coastline.mat'.format(p))
f_coastl_a_1 = np.array(f_coastl_1['Feature_coastline'])

f_alpha_1 = scipy.io.loadmat('./ID{}/Feature_frequency_alpha.mat'.format(p))
f_alpha_a_1 = np.array(f_alpha_1['Feature_frequency_alpha'])

f_delta_1 = scipy.io.loadmat('./ID{}/Feature_frequency_delta.mat'.format(p))
f_delta_a_1 = np.array(f_delta_1['Feature_frequency_delta'])

f_beta_1 = scipy.io.loadmat('./ID{}/Feature_frequency_beta.mat'.format(p))
f_beta_a_1 = np.array(f_beta_1['Feature_frequency_beta'])

f_gamma_1_1 = scipy.io.loadmat('./ID{}/Feature_frequency_gamma1.mat'.format(p))
f_gamma_1_a_1 = np.array(f_gamma_1_1['Feature_frequency_gamma1'])

f_gamma_2_1 = scipy.io.loadmat('./ID{}/Feature_frequency_gamma2.mat'.format(p))
f_gamma_2_a_1 = np.array(f_gamma_2_1['Feature_frequency_gamma2'])

f_gamma_3_1 = scipy.io.loadmat('./ID{}/Feature_frequency_gamma3.mat'.format(p))
f_gamma_3_a_1 = np.array(f_gamma_3_1['Feature_frequency_gamma3'])

f_gamma_4_1 = scipy.io.loadmat('./ID{}/Feature_frequency_gamma4.mat'.format(p))
f_gamma_4_a_1 = np.array(f_gamma_4_1['Feature_frequency_gamma4'])

f_theta_1 = scipy.io.loadmat('./ID{}/Feature_frequency_theta.mat'.format(p))
f_theta_a_1 = np.array(f_theta_1['Feature_frequency_theta'])

f_n_energy_1 = scipy.io.loadmat('./ID{}/Feature_nonlinEnergy.mat'.format(p))
f_n_energy_a_1 = np.array(f_n_energy_1['Feature_nonlinEnergy'])

f_var_1 = scipy.io.loadmat('./ID{}/Feature_variance.mat'.format(p))
f_var_a_1 = np.array(f_var_1['Feature_variance'])

p = '4_2'

### Translating mat format to numpy arrays
f_energy_2 = scipy.io.loadmat('./ID{}/Feature_Energy.mat'.format(p))
f_energy_a_2 = np.array(f_energy_2['Feature_energy'])

f_coastl_2 = scipy.io.loadmat('./ID{}/Feature_coastline.mat'.format(p))
f_coastl_a_2 = np.array(f_coastl_2['Feature_coastline'])

f_alpha_2 = scipy.io.loadmat('./ID{}/Feature_frequency_alpha.mat'.format(p))
f_alpha_a_2 = np.array(f_alpha_2['Feature_frequency_alpha'])

f_delta_2 = scipy.io.loadmat('./ID{}/Feature_frequency_delta.mat'.format(p))
f_delta_a_2 = np.array(f_delta_2['Feature_frequency_delta'])

f_beta_2 = scipy.io.loadmat('./ID{}/Feature_frequency_beta.mat'.format(p))
f_beta_a_2 = np.array(f_beta_2['Feature_frequency_beta'])

f_gamma_1_2 = scipy.io.loadmat('./ID{}/Feature_frequency_gamma1.mat'.format(p))
f_gamma_1_a_2 = np.array(f_gamma_1_2['Feature_frequency_gamma1'])

f_gamma_2_2 = scipy.io.loadmat('./ID{}/Feature_frequency_gamma2.mat'.format(p))
f_gamma_2_a_2 = np.array(f_gamma_2_2['Feature_frequency_gamma2'])

f_gamma_3_2 = scipy.io.loadmat('./ID{}/Feature_frequency_gamma3.mat'.format(p))
f_gamma_3_a_2 = np.array(f_gamma_3_2['Feature_frequency_gamma3'])

f_gamma_4_2 = scipy.io.loadmat('./ID{}/Feature_frequency_gamma4.mat'.format(p))
f_gamma_4_a_2 = np.array(f_gamma_4_2['Feature_frequency_gamma4'])

f_theta_2 = scipy.io.loadmat('./ID{}/Feature_frequency_theta.mat'.format(p))
f_theta_a_2 = np.array(f_theta_2['Feature_frequency_theta'])

f_n_energy_2 = scipy.io.loadmat('./ID{}/Feature_nonlinEnergy.mat'.format(p))
f_n_energy_a_2 = np.array(f_n_energy_2['Feature_nonlinEnergy'])

f_var_2 = scipy.io.loadmat('./ID{}/Feature_variance.mat'.format(p))
f_var_a_2 = np.array(f_var_2['Feature_variance'])

p = '4_3'

### Translating mat format to numpy arrays
f_energy_3 = scipy.io.loadmat('./ID{}/Feature_Energy.mat'.format(p))
f_energy_a_3 = np.array(f_energy_3['Feature_energy'])

f_coastl_3 = scipy.io.loadmat('./ID{}/Feature_coastline.mat'.format(p))
f_coastl_a_3 = np.array(f_coastl_3['Feature_coastline'])

f_alpha_3 = scipy.io.loadmat('./ID{}/Feature_frequency_alpha.mat'.format(p))
f_alpha_a_3 = np.array(f_alpha_3['Feature_frequency_alpha'])

f_delta_3 = scipy.io.loadmat('./ID{}/Feature_frequency_delta.mat'.format(p))
f_delta_a_3 = np.array(f_delta_3['Feature_frequency_delta'])

f_beta_3 = scipy.io.loadmat('./ID{}/Feature_frequency_beta.mat'.format(p))
f_beta_a_3 = np.array(f_beta_3['Feature_frequency_beta'])

f_gamma_1_3 = scipy.io.loadmat('./ID{}/Feature_frequency_gamma1.mat'.format(p))
f_gamma_1_a_3 = np.array(f_gamma_1_3['Feature_frequency_gamma1'])

f_gamma_2_3 = scipy.io.loadmat('./ID{}/Feature_frequency_gamma2.mat'.format(p))
f_gamma_2_a_3 = np.array(f_gamma_2_3['Feature_frequency_gamma2'])

f_gamma_3_3 = scipy.io.loadmat('./ID{}/Feature_frequency_gamma3.mat'.format(p))
f_gamma_3_a_3 = np.array(f_gamma_3_3['Feature_frequency_gamma3'])

f_gamma_4_3 = scipy.io.loadmat('./ID{}/Feature_frequency_gamma4.mat'.format(p))
f_gamma_4_a_3 = np.array(f_gamma_4_3['Feature_frequency_gamma4'])

f_theta_3 = scipy.io.loadmat('./ID{}/Feature_frequency_theta.mat'.format(p))
f_theta_a_3 = np.array(f_theta_3['Feature_frequency_theta'])

f_n_energy_3 = scipy.io.loadmat('./ID{}/Feature_nonlinEnergy.mat'.format(p))
f_n_energy_a_3 = np.array(f_n_energy_3['Feature_nonlinEnergy'])

f_var_3 = scipy.io.loadmat('./ID{}/Feature_variance.mat'.format(p))
f_var_a_3 = np.array(f_var_3['Feature_variance'])

p = '4_4'

### Translating mat format to numpy arrays
f_energy_4 = scipy.io.loadmat('./ID{}/Feature_Energy.mat'.format(p))
f_energy_a_4 = np.array(f_energy_4['Feature_energy'])

f_coastl_4 = scipy.io.loadmat('./ID{}/Feature_coastline.mat'.format(p))
f_coastl_a_4 = np.array(f_coastl_4['Feature_coastline'])

f_alpha_4 = scipy.io.loadmat('./ID{}/Feature_frequency_alpha.mat'.format(p))
f_alpha_a_4 = np.array(f_alpha_4['Feature_frequency_alpha'])

f_delta_4 = scipy.io.loadmat('./ID{}/Feature_frequency_delta.mat'.format(p))
f_delta_a_4 = np.array(f_delta_4['Feature_frequency_delta'])

f_beta_4 = scipy.io.loadmat('./ID{}/Feature_frequency_beta.mat'.format(p))
f_beta_a_4 = np.array(f_beta_4['Feature_frequency_beta'])

f_gamma_1_4 = scipy.io.loadmat('./ID{}/Feature_frequency_gamma1.mat'.format(p))
f_gamma_1_a_4 = np.array(f_gamma_1_4['Feature_frequency_gamma1'])

f_gamma_2_4 = scipy.io.loadmat('./ID{}/Feature_frequency_gamma2.mat'.format(p))
f_gamma_2_a_4 = np.array(f_gamma_2_4['Feature_frequency_gamma2'])

f_gamma_3_4 = scipy.io.loadmat('./ID{}/Feature_frequency_gamma3.mat'.format(p))
f_gamma_3_a_4 = np.array(f_gamma_3_4['Feature_frequency_gamma3'])

f_gamma_4_4 = scipy.io.loadmat('./ID{}/Feature_frequency_gamma4.mat'.format(p))
f_gamma_4_a_4 = np.array(f_gamma_4_4['Feature_frequency_gamma4'])

f_theta_4 = scipy.io.loadmat('./ID{}/Feature_frequency_theta.mat'.format(p))
f_theta_a_4 = np.array(f_theta_4['Feature_frequency_theta'])

f_n_energy_4 = scipy.io.loadmat('./ID{}/Feature_nonlinEnergy.mat'.format(p))
f_n_energy_a_4 = np.array(f_n_energy_4['Feature_nonlinEnergy'])

f_var_4 = scipy.io.loadmat('./ID{}/Feature_variance.mat'.format(p))
f_var_a_4 = np.array(f_var_4['Feature_variance'])

f_energy_a = np.concatenate((f_energy_a_1, f_energy_a_2, f_energy_a_3, f_energy_a_4), axis=1)
f_coastl_a = np.concatenate((f_coastl_a_1, f_coastl_a_2, f_coastl_a_3, f_coastl_a_4), axis=1)
f_alpha_a = np.concatenate((f_alpha_a_1, f_alpha_a_2, f_alpha_a_3, f_alpha_a_4), axis=1)
f_beta_a = np.concatenate((f_beta_a_1, f_beta_a_2, f_beta_a_3, f_beta_a_4), axis=1)
f_delta_a = np.concatenate((f_delta_a_1, f_delta_a_2, f_delta_a_3, f_delta_a_4), axis=1)
f_gamma_1_a = np.concatenate((f_gamma_1_a_1, f_gamma_1_a_2, f_gamma_1_a_3, f_gamma_1_a_4), axis=1)
f_gamma_2_a = np.concatenate((f_gamma_2_a_1, f_gamma_2_a_2, f_gamma_2_a_3, f_gamma_2_a_4), axis=1)
f_gamma_3_a = np.concatenate((f_gamma_3_a_1, f_gamma_3_a_2, f_gamma_3_a_3, f_gamma_3_a_4), axis=1)
f_gamma_4_a = np.concatenate((f_gamma_4_a_1, f_gamma_4_a_2, f_gamma_4_a_3, f_gamma_4_a_4), axis=1)
f_theta_a = np.concatenate((f_theta_a_1, f_theta_a_2, f_theta_a_3, f_theta_a_4), axis=1)
f_n_energy_a = np.concatenate((f_n_energy_a_1, f_n_energy_a_2, f_n_energy_a_3, f_n_energy_a_4), axis=1)
f_var_a = np.concatenate((f_var_a_1, f_var_a_2, f_var_a_3, f_var_a_4), axis=1)

print('TOTAL LENGTH', np.shape(f_var_a))
end = 21600
x_test = np.concatenate((f_energy_a[:,0:end],
                          f_coastl_a[:,0:end],
                          #f_alpha_a[:,0:end],
                          #f_beta_a[:,0:end],
                          #f_delta_a[:,0:end],
                          #f_gamma_1_a[:,0:end],
                          #f_gamma_2_a[:,0:end],
                          #f_gamma_3_a[:,0:end],
                          #f_gamma_4_a[:,0:end],
                          #f_theta_a[:,0:end],
                          f_n_energy_a[:,0:end],
                          f_var_a[:,0:end]), axis=0)

x_train = np.concatenate((f_energy_a[:,end:],
                          f_coastl_a[:,end:],
                          #f_alpha_a[:,end:],
                          #f_beta_a[:,end:],
                          #f_delta_a[:,end:],
                          #f_gamma_1_a[:,end:],
                          #f_gamma_2_a[:,end:],
                          #f_gamma_3_a[:,end:],
                          #f_gamma_4_a[:,end:],
                          #f_theta_a[:,end:],
                          f_n_energy_a[:,end:],
                          f_var_a[:,end:]), axis=0)  
             




p = '4_1'
labels_1 = scipy.io.loadmat('./ID{}/label_vector.mat'.format(p))
labels_a_1 = np.array(labels_1['label_vector'])
p = '4_2'
labels_2 = scipy.io.loadmat('./ID{}/label_vector.mat'.format(p))
labels_a_2 = np.array(labels_2['label_vector'])
p = '4_3'
labels_3 = scipy.io.loadmat('./ID{}/label_vector.mat'.format(p))
labels_a_3 = np.array(labels_3['label_vector'])
p = '4_4'
labels_4 = scipy.io.loadmat('./ID{}/label_vector.mat'.format(p))
labels_a_4 = np.array(labels_4['label_vector'])

labels_a = np.concatenate((labels_a_1, labels_a_2, labels_a_3, labels_a_4), axis=1)

print('length label', np.size(labels_a))
unique, counts = np.unique(labels_a, return_counts=True)
print(dict(zip(unique, counts)));

print('unique', unique)
print('counts', counts)



y_test = labels_a[0,0:end]
y_train = labels_a[0,end:]  


###############################################################
## comment these lines to avoid oversampling

x_train_seizure_1 = x_train[:,440:483]
x_train_seizure_2 = x_train[:,5755:5806]
x_train_seizure_3 = x_train[:,10813:10863]
x_train_seizure_4 = x_train[:,14287:14323]
x_train_seizure_5 = x_train[:,17167:17215]

x_train_seizure_6 = x_train[:,18000+5861:18000+5915]
x_train_seizure_7 = x_train[:,18000+9671:18000+9739]
#x_train_seizure_8 = x_train[:,18000+18410:18000+18418+1]

x_train_seizure_9 = x_train[:,21600+18000+1008:18000+21600+1043]
x_train_seizure_10 = x_train[:,21600+18000+11929:18000+21600+11959]
x_train_seizure_11 = x_train[:,21600+18000+19579:18000+21600+19609]

#ID2
print("x_train_seizure", np.shape(x_train_seizure_1))
print("x_train_seizure", np.shape(x_train_seizure_2))
print("x_train_seizure", np.shape(x_train_seizure_3))
print("x_train_seizure", np.shape(x_train_seizure_4))
print("x_train_seizure", np.shape(x_train_seizure_5))
#ID3
print("x_train_seizure", np.shape(x_train_seizure_6))
print("x_train_seizure", np.shape(x_train_seizure_7))
#print("x_train_seizure", np.shape(x_train_seizure_8))
#ID4
print("x_train_seizure", np.shape(x_train_seizure_9))
print("x_train_seizure", np.shape(x_train_seizure_10))
print("x_train_seizure", np.shape(x_train_seizure_11))


print("x_train", np.shape(x_train))
x_train = np.concatenate((x_train,
                          np.repeat(x_train_seizure_1,100,1),
                          np.repeat(x_train_seizure_2,100,1),
                          np.repeat(x_train_seizure_3,100,1),
                          np.repeat(x_train_seizure_4,100,1),
                          np.repeat(x_train_seizure_5,100,1),
                          np.repeat(x_train_seizure_6,100,1),
                          np.repeat(x_train_seizure_7,100,1),
                          #np.repeat(x_train_seizure_8,500,1),
                          np.repeat(x_train_seizure_9,100,1),
                          np.repeat(x_train_seizure_10,100,1),
                          np.repeat(x_train_seizure_11,100,1)),
                          axis=1)
print("x_train after oversampling", np.shape(x_train))
#print("x_train", x_train)        
x_train = np.transpose(x_train)
x_test = np.transpose(x_test)

print(np.shape(x_train))
print(np.shape(x_test))

  
y_train = np.concatenate((y_train, 
                          np.ones(100*43),
                          np.ones(100*51),
                          np.ones(100*50),
                          np.ones(100*36),
                          np.ones(100*48),
                          np.ones(100*54),
                          np.ones(100*68),
                          #np.ones(500*9),
                          np.ones(100*35),
                          np.ones(100*30),
                          np.ones(100*30)))
print("y_train after oversampling", np.shape(y_train))

###############################################################




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
n_connect_2 = 10 #nr of active active connections per output node (layer 1)
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
          

#          
file=h5py.File('p_acc_0.8713889122009277.h5py','r')
weight = []
for i in range(len(file.keys())):
 weight.append(file['weight'+str(i)][:])


model.set_weights(weight)
file.close()
##
##
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
index_1 = np.where(labels_a == 1)
print(index_1)
seizure_1 = np.where(index_1[1] <21600 )
#print("length 1:", np.shape(seizure_1[0]))
print('starting point', index_1[1][seizure_1[0][0]])
#print(index_1[1][seizure_1[0][-1]])

index_1 = np.where(y_pred == 1)
print(index_1[0])




