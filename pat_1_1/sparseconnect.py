# -*- coding: utf-8 -*-
"""
#################################################################################
    Paper ID: 12076
    Title: Dynamic Probabilistic Pruning: Training sparse networks based on stochastic and dynamic masking
#################################################################################
    
    Source Name   :  sparseconnect.py
    Description   :  This files contain the sparse layers and the main algorithm of 
                     Dynamic Probabilistic Pruning.

#################################################################################          
"""
import tensorflow as tf
import temperatureUpdate
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, Lambda, Activation

##########################################################################################################################
############################################################################################################################

class entropy_reg(tf.keras.regularizers.Regularizer):
    """
    Entropy penalization for trainable logits
    """

    def __init__(self, entropyMult):
        self.entropyMult = entropyMult

    def __call__(self, logits):
        normDist = tf.nn.softmax(logits,1)
        logNormDist = tf.math.log(normDist+1e-20)
        
        rowEntropies = -tf.reduce_sum(tf.multiply(normDist, logNormDist),1)
        sumRowEntropies = tf.reduce_sum(rowEntropies)
        
        multiplier = self.entropyMult
        return multiplier*sumRowEntropies

    def get_config(self):
        return {'entropyMult': float(self.entropyMult)}


    
##########################################################################################################################
############################################################################################################################
# This class includes the forward and backward part to optimize logits and generate the masks for creating the sparse net 

    
class DPS_topk(Layer):
    
    """
    - DPS_topK optimizes logits, and gets k samples from each categorical distribution 
    - It returns hardSamples during forwardpass
    - It uses SoftSamples during backward pass
    """
    
    
    def __init__(self,BS, k, batchPerEpoch=1, n_epochs=2,  tempIncr=5, name=None,**kwargs):
        self.BS = BS                     # The dynamic batch_size parameter
        self.k = k                       # Define the number of weights per output node that should be used. (k <= input_nodes)
        self.batchPerEpoch = batchPerEpoch  # Amount of batch updates per epoch
        self.n_epochs = n_epochs            # Total number of epochs used for training
        self.tempIncr = tempIncr        # Value with which the temperature in the softmax is multiplied at the beginning of training
        
        #self.outShape = (self.BS, self., self.k, self.input_nodes)

        super(DPS_topk, self).__init__(name=name,**kwargs) 

    def build(self, input_shape):
        self.step = K.variable(0)
        super(DPS_topk, self).build(input_shape)  
      
    def call(self, inp):
        print('inp shape', inp.shape)
        logits = inp  #[output_nodes,input_nodes]


        data_shape = tf.shape(inp)
        
        ### Forwards ###
        GN = -tf.math.log(-tf.math.log(tf.random.uniform((self.BS,data_shape[0], data_shape[1]),0,1)+1e-20)+1e-20) #[BS,output_nodes,input_nodes]
        perturbedLog = logits+GN #[BS,output_nodes,input_nodes]
        
        # Find the top-k indices. Apply top_k second time to sort them from high to low
        ind =  tf.nn.top_k(tf.nn.top_k(perturbedLog, k=self.k).indices,k=self.k).values  #[BS,output_nodes,k]

        # Reverse the sorting to have the indices from low to high
        topk = tf.reverse(tf.expand_dims(ind,-1), axis=[2]) #[BS,output_nodes,k]
       
        hardSamples = tf.squeeze(tf.one_hot(topk,depth=data_shape[-1]),axis=-2) #[BS,output_nodes,k,input_nodes]
      
        ### Backwards ###
        # This is the softamax function, which is used during training as a relaxation function
        epoch = self.step/self.batchPerEpoch
        Temp = temperatureUpdate.temperature_update_tf(self.tempIncr, epoch, self.n_epochs)
        updateSteps = []
        updateSteps.append((self.step, self.step+1))
        self.add_update(updateSteps,inp)

        prob_exp = tf.tile(tf.expand_dims(tf.expand_dims(tf.exp(logits),0),2),(self.BS,1,self.k,1)) #[BS,output_nodes,k,input_nodes]  
        cumMask = tf.cumsum(hardSamples,axis=-2, exclusive=True) #[BS,output_nodes,k,input_nodes]
        softSamples = tf.nn.softmax((tf.math.log(tf.multiply(prob_exp,1-cumMask+1e-20))+tf.tile(tf.expand_dims(GN,-2),(1,1,self.k,1)))/Temp, axis=-1)  #[BS,output_nodes,k,input_nodes]

        return tf.stop_gradient(hardSamples - softSamples) + softSamples
        
    

    
##########################################################################################################################
############################################################################################################################
class sparseconnect_layer(Layer):  
    
    """
    Sparse fully-connected layer
    - Generates trainable logits (D)
    - Call DPS_topK to perform optimization
    - Generates a mask based on hardSamples to sparsify W matrix
    """
    
    def __init__(self,units, n_connect, activation=None, n_epochs=10, tempIncr=5, name=None, one_per_batch=True):
        self.units = units
        self.n_connect = n_connect
        self.activation = activation
        self.n_epochs = n_epochs
        self.tempIncr = tempIncr
        self.one_per_batch = one_per_batch 
        super(sparseconnect_layer, self).__init__(name=name) 

    def build(self, input_shape): 
        # Define weight matrix and bias vector
        self.W = self.add_weight(shape=[self.units,int(input_shape[-1])],
                                 initializer='glorot_uniform',
                                 trainable=True, name='w_bin', dtype='float32')
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='glorot_normal',
                                 trainable=True, name='b_bin',  dtype='float32')
        
        # Define sampling logits
        self.D = self.add_weight(name='TrainableLogits', 
                              shape=(self.units, input_shape[-1]),
                              initializer = tf.random_normal_initializer(mean=0, stddev=0.5, seed=None),
                              regularizer = entropy_reg(5e-3),
                              #initializer = tf.initializers.RandomNormal(minval=-1.0, maxval=1.0, seed=None), #TODO Liz: choose which initializer might be suitable for you
                              trainable=True)

        super(sparseconnect_layer, self).build(input_shape) 
 
    def call(self, x):
        units = self.units

        if self.one_per_batch:
            batch_size = K.shape(x)[0]
            # Produce sparse sampling matrix
            A = DPS_topk(BS=1, k = self.n_connect, n_epochs=self.n_epochs,  tempIncr=self.tempIncr)(self.D)
            A = tf.reduce_sum(A,axis=-2)
            A = tf.reduce_sum(A,axis=0)            
            # Produce sparse weight matrix
            AW = Lambda(lambda inp: tf.multiply(inp[0],inp[1]), output_shape = (units,units))([A,self.W])  
            # Produce layer output
            y = Lambda(lambda inp: K.dot(inp[1],tf.transpose(inp[0],(1,0)))+inp[2], output_shape = (units))([AW,x,self.b])
            #y = tf.keras.layers.BatchNormalization()(y)            
        else:
            batch_size = K.shape(x)[0]
            # Produce sparse sampling matrix
            A = DPS_topk(BS=batch_size, k = self.n_connect, n_epochs=self.n_epochs,  tempIncr=self.tempIncr)(self.D)
            A = tf.reduce_sum(A,axis=-2)          
            # Produce sparse weight matrix
            AW = Lambda(lambda inp: tf.multiply(inp[0],inp[1]), output_shape = (units,units))([A,self.W])  
            # Produce layer output
            y = Lambda(lambda inp: K.batch_dot(inp[1],tf.transpose(inp[0],(0,2,1)))+inp[2], output_shape = (units))([AW,x,self.b])
        if not self.activation == None:
            y = Activation(self.activation)(y)
        
        return y

