# Copyright (c) 2022, bochi
# All rights reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# * Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# * Neither the name of the maebashi university nor the names of its contributors
# may be used to endorse or promote products derived from this software
# without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED

import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import random
import peek_pridict
import fft_wave

file_name = [
    "20230210_toyama_safe.csv", #0
    "20230327_toyama_safe.csv", #1
    "20230327_sekiyama_safe.csv", #2
    "20230522_masuda_newsensa.csv", #3
    "20230523_toyama_newsensa.csv", #4
    "A_deskwork_500Hz.csv", #5
    "B_deskwork_500Hz.csv", #6
    "C_deskwork_500Hz.csv", #7
    "E_deskwork_500Hz.csv", #8
    "c_deskwork_500Hz(filter).csv", #9
    "20230530_masuda_hard.csv", #10
    "20230530_masuda_loose.csv", #11
    "20230530_toyama_hard.csv", #12
    "20230530_toyama_loose.csv", #13
    "500Hz_F_60.csv", #14
    "500Hz_G_60.csv", #15
    "20230616_bochi_safe3point.csv", #16
    "20230616_masuda_safe3point.csv", #17
    "20230621_bochi_safe3point.csv", #18
    "20230621_masuda_safe3point.csv", #19
]


# 波形の表示
def peak_detect(x,y,lab,ID):
    fig = plt.figure()
    fig.set_figwidth(20)
    plt.plot(x, y, label=lab, c="blue", alpha=0.5)
    plt.plot(x[ID], y[ID], "ro")
    plt.xlabel("time")
    plt.ylabel("ECG")
    plt.legend(loc="upper right")
    plt.show()

def moving_diff(ecg, range1, range2):
    fs = 500
    period = 600
    t = np.arange(0, period, 1/fs)
    ecg_time = list(range(len(ecg) - 11))
    points = 10
    moving_ave = []
    for i in range(0, len(ecg) - points):
        tmp = 0
        for j in range(0, points):
            tmp += ecg[i + j]

        tmp /= points
        moving_ave.append(tmp)
    moving_ave = np.diff(moving_ave)
    # fig1 = plt.figure()
    # fig1.set_figwidth(20)
    # plt.plot(ecg_time[range1:range2], moving_ave[range1:range2], label="ECG")
    # plt.show()

    time = np.array(ecg_time[range1:range2])
    wave = np.array(moving_ave[range1:range2])
    
    return time, wave

df_data1 = pd.read_csv('wave_file/20220623_bochi_gaming.csv', usecols=[3,2], names=('heart', 'pulse'))
df_Tdata1 = df_data1.T
# df_data2 = pd.read_csv('wave_file/company/20220606_toyama_2.csv', usecols=[1,2], names=('heart', 'pulse'))
# df_Tdata2 = df_data2.T
# df_data3 = pd.read_csv('wave_file/company/20220606_toyama_3.csv', usecols=[1,2], names=('heart', 'pulse'))
# df_Tdata3 = df_data3.T

np_pulse = np.array(df_data1['pulse'][0:300000])
np_heart = np.array(df_data1['heart'][0:300000])
diff_heart = moving_diff(np_heart, 0, 300000)
time = diff_heart[0]
diff_heartwave = diff_heart[1]
maxid = signal.argrelmax(diff_heartwave, order = 300)
time_stamp1 = diff_heart[0][maxid]
# peak_detect(time,diff_heartwave,"ECG",maxid)

# np_pulse = np.array(df_data2['pulse'][0:300000])
# np_heart = np.array(df_data2['heart'][0:300000])
# diff_heart = moving_diff(np_heart, 0, 300000)
# time = diff_heart[0]
# diff_heartwave = diff_heart[1]
# maxid = signal.argrelmax(diff_heartwave, order = 300)
# time_stamp2 = diff_heart[0][maxid]
# # peak_detect(time,diff_heartwave,"ECG",maxid)

# np_pulse = np.array(df_data3['pulse'][0:300000])
# np_heart = np.array(df_data3['heart'][0:300000])
# diff_heart = moving_diff(np_heart, 0, 300000)
# time = diff_heart[0]
# diff_heartwave = diff_heart[1]
# maxid = signal.argrelmax(diff_heartwave, order = 300)
# time_stamp3 = diff_heart[0][maxid]
# # peak_detect(time,diff_heartwave,"ECG",maxid)


x = np.array([])
y = np.array([])

def read_data(time, data, x, y):
    true = pd.DataFrame([1])
    false = pd.DataFrame([0])
    for i in range(1, len(time)):
        pulse_on_time = 76
        start_time = int(time[i-1])
        midle1_time = int(time[i-1] + pulse_on_time)
        midle2_time = int(time[i-1] + pulse_on_time*2)
        midle3_time = int(time[i-1] + pulse_on_time*3)
        end_time = int(time[i-1] + pulse_on_time*4)

        #start_time
        x = np.append(x,data.loc['pulse',start_time:midle1_time].values)
        y = np.append(y,false.loc[0].values)

        x = np.append(x,data.loc['pulse',midle1_time:midle2_time].values)
        y = np.append(y,true.loc[0].values)

        x = np.append(x,data.loc['pulse',midle2_time:midle3_time].values)
        y = np.append(y,false.loc[0].values)
        x = np.append(x,data.loc['pulse',midle3_time:end_time].values)
        y = np.append(y,false.loc[0].values)

        #half separete
        # x = np.append(x,data.loc['pulse',start_time:midle2_time].values)
        # y = np.append(y,true.loc[0].values)
        # x = np.append(x,data.loc['pulse',midle2_time:end_time].values)
        # y = np.append(y,false.loc[0].values)
    
    return x, y

x = read_data(time_stamp1, df_Tdata1, x, y)[0]
# x = read_data(time_stamp2, df_Tdata2, x, y)[0]
# x = read_data(time_stamp3, df_Tdata3, x, y)[0]
y = read_data(time_stamp1, df_Tdata1, x, y)[1]
# y = read_data(time_stamp2, df_Tdata2, x, y)[1]
# y = read_data(time_stamp3, df_Tdata3, x, y)[1]
# x = np.array(x).reshape(10516,76)  #toyama shape
# x = np.array(x).reshape(3428,76)  #toyama shape (10minutes)
# x = np.array(x).reshape(5258,153)  #toyama halfshape
# x = np.array(x).reshape(11040,76)  #iwai shape
# x = np.array(x).reshape(4888, 153)  #iwai halfshape
# x = np.array(x).reshape(11040,76)  #takahashi shape
# x = np.array(x).reshape(4686,153)  #takahashi halfshape
# x = np.array(x).reshape(8088,76)  #iida shape
# x = np.array(x).reshape(4044,153)  #iida halfshape
# x = np.array(x).reshape(6244,76)  #wan shape
# x = np.array(x).reshape(3122,153)  #wan halfshape
# x = np.array(x).reshape(2896,81)  #bochi/0622safe shape
# x = np.array(x).reshape(2740,77)  #bochi/0622move shape
# x = np.array(x).reshape(1370,153)  #bochi/0622move halfeshape
x = np.array(x).reshape(1988,77)  #bochi/0623move shape
# x = np.array(x).reshape(1192,153)  #bochi/0623move halfshape
# x = np.array(x).reshape(1134,179)  #bochi/1128cushion halfshape
# x = np.array(x).reshape(2268,90)  #bochi/1128cushion quotershape
# x = np.array(x).reshape(1240,171)  #bochi/1127haritsuke halfshape
# x = np.array(x).reshape(2480,86)  #bochi/1127haritsuke quotershape
print(x.shape)
print(y.shape)

# データを学習用とテスト用に分割する
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle = True)


#k近傍による学習
from sklearn import neighbors
n_neighbors =5
clf_result=neighbors.KNeighborsClassifier(n_neighbors, weights = 'distance') 
#学習モデル
clf_result.fit(x_train, y_train)

#学習モデルの保存
filename = 'safepulse_model.sav'
pickle.dump(clf_result, open(filename, 'wb'))


y_pred = clf_result.predict(x_test)
print(accuracy_score(y_test, y_pred))
fig = plt.figure()
fig.set_figwidth(50)
ax1 = fig.add_subplot(3, 1, 1)
ax2 = fig.add_subplot(3, 1, 2)
ax3 = fig.add_subplot(3, 1, 3)
ax1.plot(np.arange(len(y_test)), y_test)
ax2.plot(np.arange(len(y_pred)), y_pred)
ax3.plot(time, diff_heartwave, label='ECG', c="blue", alpha=0.5)
ax3.plot(time[maxid], diff_heartwave[maxid], "ro")
plt.show()

#混同行列、正解率、適合率、再現率、F値を表示
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score,recall_score,f1_score
print('混同行列 = \n', confusion_matrix(y_true = y_test, y_pred = y_pred))
print('正解率 = ',accuracy_score(y_true = y_test , y_pred = y_pred))
print('適合率 = ',precision_score(y_true = y_test , y_pred = y_pred))
print('再現性 = ',recall_score(y_true = y_test , y_pred = y_pred))
print('f1値 = ',f1_score(y_true = y_test , y_pred = y_pred))

from sklearn.metrics import log_loss
print('log loss = ', log_loss(y_true=y_test,y_pred=y_pred))