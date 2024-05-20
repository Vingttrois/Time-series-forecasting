# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 14:49:33 2021

@author: FanW
"""

from keras.layers import LSTM, RepeatVector, Dense, Dot, Input,\
     Activation, Add, Reshape, Lambda, Multiply, Concatenate, Layer, TimeDistributed
from keras.models import Model, Sequential



#N vs N结构
def NvN_LSTM(n_h, n_steps_in, n_features_in, n_features_out):
    
    model = Sequential()
    model.add(LSTM(n_h, activation='relu', input_shape=(n_steps_in, n_features_in), return_sequences=True))
    model.add(TimeDistributed(Dense(n_features_out)))

    return model


#N vs 1结构
def Nv1_LSTM(n_h, n_steps_in, n_features_in, n_features_out):
    
    model = Sequential()
    model.add(LSTM(n_h, activation='relu', input_shape=(n_steps_in, n_features_in)))
    model.add(Dense(n_features_out))
    
    return model

#堆叠LSTM
def Stack_LSTM(n_h, n_steps_in, n_features_in, n_features_out):
    
    model = Sequential()
    model.add(LSTM(n_h, activation='relu', input_shape=(n_steps_in, n_features_in), return_sequences=True))
    model.add(LSTM(n_h, activation='relu'))
    model.add(Dense(n_features_out))
    
    return model