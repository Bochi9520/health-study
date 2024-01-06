# gpu認識テスト
# from tensorflow.python.client import device_lib
# device_lib.list_local_devices()

import fft_wave
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wave_study as ws
import pickle
import peek_pridict
from scipy import signal
import time

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
]

before_start = time.time()
n = int(input('使用するファイルの番号を入力してください:'))
heart, pulse = fft_wave.read_data(n)
pulse = fft_wave.lowpass(pulse, 5, 20, 500)

t = np.arange(0, 600, 1/500)
showrange_bottom = 150000
showrange_top = 158000
test_pulse, test_heart, test_data = ws.create_windowdata(
    pulse[showrange_bottom:showrange_top], heart[showrange_bottom:showrange_top], 200) #検証用脈波、検証用心電、検証用窓変換脈波
filename = 'wave_study.sav'
model_load = pickle.load(open(filename, 'rb'))
pridict = model_load.predict(test_data)
# pridict = fft_wave.lowpass(pridict, 5, 20, 500)
befoer_end = time.time()
print("学種済みを利用した際の処理時間:"+ str(befoer_end - before_start))

pulse = np.array(pulse)
pulse_id = np.array(peek_pridict.get_peaks_pulse(pulse[showrange_bottom:showrange_top], 1.0, 0, 8000))
# pulse_id = signal.argrelmax(pulse[showrange_bottom:showrange_top], order=200)
time_pulse = np.array(ws.pulsepeek_piucture(pulse[showrange_bottom:showrange_top], '脈波', pulse_id))
fpulse_id = np.array(peek_pridict.get_peaks_pulse(pridict, -0.002, 0, len(pridict)))
# fpulse_id = signal.argrelmax(pridict, order=200)
time_fpulse = np.array(ws.pulsepeek_piucture(pridict, '疑似脈波', fpulse_id))

fig = plt.figure()
fig.set_figwidth(20)
fig.set_figheight(10)
plt.plot(t[showrange_bottom:showrange_top], pulse[showrange_bottom:showrange_top], "b", alpha=0.5)
plt.plot(t[showrange_bottom:showrange_top][pulse_id], pulse[showrange_bottom:showrange_top][pulse_id], "ro")
plt.show()
fig = plt.figure()
fig.set_figwidth(20)
fig.set_figheight(10)
plt.plot(t[showrange_bottom:showrange_bottom+len(pridict)], pridict, "b", alpha=0.5)
plt.plot(t[showrange_bottom:showrange_bottom+len(pridict)][fpulse_id], pridict[fpulse_id], "ro")
plt.show()

lendiff = len(time_pulse)-len(time_fpulse)
if len(time_pulse)-len(time_fpulse) < 0:
    for i in range(lendiff):
        time_pulse = np.append(time_pulse, None)
if len(time_pulse)-len(time_fpulse) > 0:
    for i in range(lendiff):
        time_fpulse = np.append(time_fpulse, None)

databox = pd.DataFrame(index=[], columns=['脈拍時刻', '疑似脈拍時刻'])
databox = databox.assign(脈拍時刻=time_pulse)
databox = databox.assign(疑似脈拍時刻 = time_fpulse)
databox.to_csv('data/testdata_pd.csv', encoding='utf_8_sig')

# heart_id = signal.argrelmax(heart, order=200)
# heart_diff = np.diff(heart)
# heartdiff_id = signal.argrelmax(heart_diff, order=200)
# fig = plt.figure()
# fig.set_figwidth(20)
# plt.plot(t[0:299999], heart_diff, c="blue", alpha=0.5)
# plt.plot(t[heartdiff_id], heart_diff[heartdiff_id], "ro")
# plt.show()

