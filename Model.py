# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 14:49:33 2021

@author: FanW
"""


import numpy as np
from keras.layers import LSTM, RepeatVector, Dense, Dot, Input,\
      Activation, Add, Reshape, Lambda, Multiply, Concatenate, Layer, TimeDistributed
from keras.models import Model, Sequential


def split_sequences(sequences, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequences)):
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        if out_end_ix > len(sequences):
            break
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix:out_end_ix, -1]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)



##简单Encoder-Decoder架构
def ED_LSTM(n_h_en, n_h_de, n_steps_in, n_steps_out, n_features_in, n_features_out, mark):
    
    if mark == 'Sequential':
        '通过Sequential方法构建'
        model = Sequential()
        model.add(LSTM(n_h_en, activation='relu', input_shape=(n_steps_in, n_features_in))) #传递的是hidden_state, 如Sutskever的做法
        model.add(RepeatVector(n_steps_out))
        model.add(LSTM(n_h_de, activation='relu', return_sequences=True))
        model.add(TimeDistributed(Dense(n_features_out)))

    if mark == 'Model':
        '通过Model方法构建'
        inputs_Encoder = Input(shape=(n_steps_in, n_features_in))
        Encoder_LSTM = LSTM(n_h_en, activation='relu', return_state=True) #return_state, 返回c
        outputs_Encoder, state_h, state_c = Encoder_LSTM(inputs_Encoder)
        inputs_Context = RepeatVector(n_steps_out, )(state_c)
        Decoder_LSTM = LSTM(n_h_de, activation='relu', return_sequences=True, return_state=True)
        outputs_Decoder, _, _ = Decoder_LSTM(inputs_Context)
        Decoder_Dense = Dense(n_features_out, activation='relu')
        outputs_Decoder = Decoder_Dense(outputs_Decoder)
        model = Model(inputs_Encoder, outputs_Decoder)

    return model






