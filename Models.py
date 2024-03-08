# This software is Copyright © 2024 The Regents of the University of California. 
# All Rights Reserved. Permission to copy, modify, and distribute this software and 
# its documentation for educational, research and non-profit purposes, without fee, and 
# without a written agreement is hereby granted, provided that the above copyright notice, 
# this paragraph and the following three paragraphs appear in all copies. Permission to 
# make commercial use of this software may be obtained by contacting:
# Office of Innovation and Commercialization
# 9500 Gilman Drive, Mail Code 0910
# University of California
# La Jolla, CA 92093-0910
# innovation@ucsd.edu
# This software program and documentation are copyrighted by The Regents of the University of California. 
# The software program and documentation are supplied “as is”, without any accompanying services 
# from The Regents.The Regents does not warrant that the operation of the program will be 
# uninterrupted or error-free.The end-user understands that the program was developed for 
# research purposes and is advised not to rely exclusively on the program for any reason.
# IN NO EVENT SHALL THE UNIVERSITY OF CALIFORNIA BE LIABLE TO ANY PARTY FOR DIRECT, 
# INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, 
# ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF 
# THE UNIVERSITY OF CALIFORNIA HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. 
# THE UNIVERSITY OF CALIFORNIA SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT 
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. 
# THE SOFTWARE PROVIDED HEREUNDER IS ON AN “AS IS” BASIS, AND THE UNIVERSITY OF CALIFORNIA 
# HAS NO OBLIGATIONS TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.



#!/usr/bin/python3

#Author: Vaghef Ghazvinian, mghazvinian@ucsd.edu
#Affiliation: CW3E, Scripps, UCSD





import tensorflow as tf
import tensorflow.keras.backend as k
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.layers import Embedding, Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv1D, Conv1DTranspose, MaxPool1D, AveragePooling1D
from tensorflow.keras.layers import SpatialDropout1D, Dropout
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.losses import CategoricalCrossentropy
from losses import *
import numpy as np


class Terminateonnan(tf.keras.callbacks.Callback):
  """Callback that terminates training when a NaN loss is encountered.
  """
  def __init__(self):
    super(Terminateonnan, self).__init__()
    self._supports_tf_logs = True

  def on_epoch_end(self, epoch,logs=None):
    logs = logs or {}
    loss = logs.get('loss')
    val_loss=logs.get("val_loss")
    if (epoch==0 and np.isnan(loss)) or (epoch==0 and np.isinf(val_loss)):
        print('epoch %d: Invalid loss, terminating training' % (epoch))
        self.model.stop_training = True






def build_csgd_embed_model(n_features=6,embedding_dim = None, embedding_size=25,hidden_nodes=[10],
                           activation='softplus',loss = crps_cens,
                           optimizer="Adam",lr=0.01 ,par_reg=1e-5,conv=True):
    """This method builds keras model with locational 
    embedding and convolutional layer, outputs the 
    compiled model 
    """ 
    inputs = []
    loc_input = Input(shape=[1], name='location')
    inputs.append(loc_input)
    loc_embedding = Embedding(embedding_dim,embedding_size)(loc_input)
    numerical_inputs = Input(shape=[n_features], name='numericals')
    inputs.append(numerical_inputs)
    reshaped_numericals = Reshape((1, numerical_inputs.shape[1]))(numerical_inputs)
    layer = concatenate([loc_embedding, reshaped_numericals], axis=2)   
    layer = Flatten()(layer)
    if conv:
        layer = Reshape((layer.shape[1], 1))(layer)
        layer = Conv1D(filters=3, kernel_size=3, activation='softplus')(layer)
        layer = MaxPool1D(pool_size=2)(layer)  
        layer = Flatten()(layer) 
        
    x = Dense(hidden_nodes[0], activation=activation,kernel_regularizer=l1(par_reg))(layer)
    if len(hidden_nodes) > 1:
        for h in hidden_nodes[1:]:
            x = Dense(h,activation=activation,kernel_regularizer=l1(par_reg))(x)               
    x = Dense(3, activation="softplus")(x)
    model= Model(inputs=inputs, outputs=x)
    opt = tf.keras.optimizers.__dict__[optimizer](learning_rate=lr)
    model.compile(loss = loss, optimizer = opt)
    return model
    
 
 
def train_csgd_embed_model(model,train_x,train_y,valid_x,valid_y,
                           batch_size=10000, epochs=1000):
    """This method receives a keras model and train and validation data, and
    trains the model based on the input arguments.
    Returns full training history.
    """ 
    newX_t = {}
    newX_t['location'] = train_x[:,-1].astype(np.int32)
    newX_t['numericals'] = np.delete(train_x,-1,axis=1).astype(np.float32)
    
    newX_v = {}
    newX_v['location'] = valid_x[:,-1].astype(np.int32)
    newX_v['numericals'] = np.delete(valid_x,-1,axis=1).astype(np.float64)
     
    # Train the model.
    loss_history = model.fit(newX_t ,train_y,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,shuffle=True,
        validation_data = (newX_v,valid_y),
        callbacks=[EarlyStopping(
            monitor='val_loss',
            min_delta=1e-6,
            patience=5,
            restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', 
            factor=0.2,patience=5, 
            min_lr=0.00001,verbose=1),Terminateonnan()])     
    # Return model's history
    return loss_history.history
        
  











