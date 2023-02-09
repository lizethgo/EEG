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


### Translating mat format to numpy arrays
f_energy = scipy.io.loadmat('./ID1/Feature_Energy.mat')
f_energy_a = np.array(f_energy['Feature_energy'])

f_coastl = scipy.io.loadmat('./ID1/Feature_coastline.mat')
f_coastl_a = np.array(f_coastl['Feature_coastline'])

f_alpha = scipy.io.loadmat('./ID1/Feature_frequency_alpha.mat')
f_alpha_a = np.array(f_alpha['Feature_frequency_alpha'])

f_delta = scipy.io.loadmat('./ID1/Feature_frequency_delta.mat')
f_delta_a = np.array(f_delta['Feature_frequency_delta'])

f_beta = scipy.io.loadmat('./ID1/Feature_frequency_beta.mat')
f_beta_a = np.array(f_beta['Feature_frequency_beta'])

f_gamma_1 = scipy.io.loadmat('./ID1/Feature_frequency_gamma1.mat')
f_gamma_1_a = np.array(f_gamma_1['Feature_frequency_gamma1'])

f_gamma_2 = scipy.io.loadmat('./ID1/Feature_frequency_gamma2.mat')
f_gamma_2_a = np.array(f_gamma_2['Feature_frequency_gamma2'])

f_gamma_3 = scipy.io.loadmat('./ID1/Feature_frequency_gamma3.mat')
f_gamma_3_a = np.array(f_gamma_3['Feature_frequency_gamma3'])

f_gamma_4 = scipy.io.loadmat('./ID1/Feature_frequency_gamma4.mat')
f_gamma_4_a = np.array(f_gamma_4['Feature_frequency_gamma4'])

f_theta = scipy.io.loadmat('./ID1/Feature_frequency_theta.mat')
f_theta_a = np.array(f_theta['Feature_frequency_theta'])

f_n_energy = scipy.io.loadmat('./ID1/Feature_nonlinEnergy.mat')
f_n_energy_a = np.array(f_n_energy['Feature_nonlinEnergy'])

f_var = scipy.io.loadmat('./ID1/Feature_variance.mat')
f_var_a = np.array(f_var['Feature_variance'])

data_set = np.concatenate((f_energy_a[0:],
                          f_coastl_a[0:],
                          f_alpha_a[0:],
                          f_beta_a[0:],
                          f_delta_a[0:],
                          f_gamma_1_a[0:],
                          f_gamma_2_a[0:],
                          f_gamma_3_a[0:],
                          f_gamma_4_a[0:],
                          f_theta_a[0:],
                          f_n_energy_a[0:],
                          f_var_a[0:]), axis=0)

labels = scipy.io.loadmat('./ID1/label_vector.mat')
labels_a = np.array(labels['label_vector'])


unique, counts = np.unique(labels_a, return_counts=True)
print(dict(zip(unique, counts)));
## there are 1220 1's againts 20380 0's

## index_1 will store the values of indexes when label is 1
## index [1] stores the actual indexes of 1's
index_1 = np.where(labels_a == 1)

## First sizure (array length) --> 620 values (starting at 3959 - 4579)
seizure_1 = np.where(index_1[1] < 10000)

## Second sizure (array length) --> 600 values (starting at 15096 - 15696)
seizure_2 = np.where(index_1[1] > 10000)




