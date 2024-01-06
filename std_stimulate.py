import fft_wave
import peek_pridict

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import pandas as pd

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
    "safe_ito_20230801.csv", #20
]

n = int(input('使用するファイルの番号を入力してください:'))
fs = 500
heart, pulse = fft_wave.read_data(n, fs, 1)
# heart, pulse = fft_wave.read_data(n, fs, 2)
t = np.arange(0, 600, 1/fs)

zscore_heart = scipy.stats.zscore(heart)
zscore_pulse = scipy.stats.zscore(pulse)
pulse = fft_wave.lowpass(pulse, 5, 20, fs)
new_pulse = fft_wave.lowpass(zscore_pulse, 5, 7, fs)
databox = pd.DataFrame(index=[], columns=['標準化heart', '標準化pulse'])
databox = databox.assign(標準化heart=zscore_heart)
databox = databox.assign(標準化pulse=new_pulse)
databox.to_csv("wave_file/zscore.csv", encoding='shift_jis')


fig = plt.figure()
fig.set_figwidth(20)
fig.set_figheight(10)
# new_pulse_id = np.array(peek_pridict.get_peaks_pulse(new_pulse, ))
plt.plot(t[5000:10000], pulse[5000:10000], "r", alpha=0.5)
plt.plot(t[5000:10000], new_pulse[5000:10000], "b", alpha=0.5)
# plt.plot(t[pulse_id], pulse[showrange_bottom:showrange_top][pulse_id], "ro")
plt.show()

# fft_wave.fft_image(pulse, 90, 300, n)
fft_wave.fft_image(new_pulse, 50, 300, n)