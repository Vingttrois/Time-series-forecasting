# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 15:15:18 2022

@author: FanW
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib as mpl
import json
import Model


#Read Input Data
Data_file = open("json1.json", 'r', encoding='utf8')
Data_data = json.load(Data_file)
Data_df = pd.DataFrame(Data_data)
Data_file.close()

Z_R = list(Data_df['Z'])[0]
P1_R = list(Data_df['P1'])[0]
P2_R = list(Data_df['P2'])[0]
P3_R = list(Data_df['P3'])[0]
P4_R = list(Data_df['P4'])[0]
P5_R = list(Data_df['P5'])[0]
P6_R = list(Data_df['P6'])[0]
P7_R = list(Data_df['P7'])[0]
P8_R = list(Data_df['P8'])[0]

zz_list = np.array(Z_R)
pp_list = []
pp_list.append(P1_R)
pp_list.append(P2_R)
pp_list.append(P3_R)
pp_list.append(P4_R)
pp_list.append(P5_R)
pp_list.append(P6_R)
pp_list.append(P7_R)
pp_list.append(P8_R)

in_list = []
for i in range(0, len(pp_list)):
    in_list.append(np.array(pp_list[i]))

dataset = zz_list.reshape((len(zz_list), 1))
for i in range(0, len(in_list)):
    dataset = np.hstack((dataset, in_list[i].reshape((len(in_list[i]), 1))))
dataset = np.hstack((dataset, zz_list.reshape((len(zz_list), 1))))


#Generate Sample
n_steps_in = 6 #Length of Input Sequence
n_steps_out = 1 #Length of Onput Sequence
X_set, y_set = Model.split_sequences(dataset, n_steps_in, n_steps_out)

#Parameters
n_features_in = X_set.shape[2]
n_features_out = 1
X_train = X_set
y_train = np.reshape(y_set,(y_set.shape[0],y_set.shape[1],1))   
n_h_en, n_h_de = 32, 32


mark = 'Sequential'
# mark = 'Model'

model = Model.ED_LSTM(n_h_en, n_h_de, n_steps_in, n_steps_out, n_features_in, n_features_out, mark)

#Traning
model.compile(optimizer='adam', loss='mse')
model.summary()


history = model.fit(X_train, y_train, epochs=100, batch_size = 512, verbose=1, shuffle=True, validation_split = 0.1) 


plt.figure()
plt.plot(history.history['loss'],'b')
plt.plot(history.history['val_loss'],'r')
# model.save_weights('s2s.h5')

# model.load_weights("s2s.h5") #加载模型

#Prediction
predict_list = []
for i in range(0, len(X_set)):
    print(i, len(X_set))
    X_input = X_set[i]
    X_input = np.reshape(X_input, (1, X_input.shape[0], X_input.shape[1]))
    y_pred = model.predict(X_input, verbose=0)
    predict_list.append(y_pred[0][0][0])

obs_list = []
for i in range(0, len(y_train)):
    obs_list.append(y_train[i][0][0])


def num2color(values, cmap):
    norm = mpl.colors.Normalize(vmin=np.min(values), vmax=np.max(values))
    cmap = mpl.cm.get_cmap(cmap)
    return [cmap(norm(val)) for val in values]
colors = num2color(predict_list, "turbo")

plt.rc('font',family='Times New Roman')
fontcn = {'family': 'STSong'}

plt.figure(figsize=(16, 6))
plt.plot(obs_list,'k--')
plt.scatter(np.arange(0,len(predict_list)), np.array(predict_list), color = colors)

timeid_list = np.linspace(0, 52554, 12)
plt.xticks(timeid_list, ['2020-01','2020-02','2020-03','2020-04','2020-05',\
                         '2020-06','2020-07','2020-08','2020-09','2020-10','2020-11','2020-12'],size = 16)

plt.yticks(size = 16)
plt.ylabel(u'流量($\mathregular{m^{3}}$/s)', fontcn, size = 16)
plt.xlabel(u'时间', fontcn, size = 16)
plt.subplots_adjust(left=0.05, bottom=0.1, right=0.99, top=0.98)


























