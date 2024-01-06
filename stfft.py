import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
import fft_wave
from scipy import stats
import time

#正規化を行う関数の定義
def min_max_df(df):
    #最小値の計算
    df_Zxx_min = df.min()
    #最大値の計算
    df_Zxx_max = df.max()
    #正規化の計算
    min_max_df = (df - df_Zxx_min) / (df_Zxx_max - df_Zxx_min)
    return min_max_df

#初期値設定
fs = 500
N = 600
t = np.arange(0, 600, 1/500)
step = 2500

#データ読み込み
n = int(input('ファイル番号を指定してください:'))
ecg_heart, ecg_pulse = fft_wave.read_data(n, fs, 1) #silver
# ecg_heart, ecg_pulse = fft_wave.read_data(n, fs, 2) #file
# ecg_heart = fft_wave.lowpass(ecg_heart, 5, 20, 500)
ecg_pulse = fft_wave.lowpass(ecg_pulse, 5, 30, 500)
ecg_pulse = fft_wave.highpass(ecg_pulse, 2, 0.5, 500)

freq, ts, Zxx = signal.stft(ecg_pulse, fs, nperseg=500, noverlap=350)
print(freq.shape)
print(ts.shape)
#Zxx: freq × ts, Zxx.T: ts × freq
# print((np.abs(Zxx))[6])
# print(np.average((np.abs(Zxx.T))[6]))

# スペクトログラム分析の実施
for i in range(0, 297500, step):
    # f, t, Sxx = signal.spectrogram(ecg_pulse[25000: 35000], fs, 100)
    fft_data = np.fft.fft(ecg_pulse[i:i+step])
    freq = np.fft.fftfreq(len(ecg_pulse[i:i+step]), 1/fs)
    Amp = abs(fft_data/(len(ecg_pulse[i:i+step])/2))
    #図の描画
    # plt.pcolormesh(time[i:i+2500], freq, 10*np.log(Sxx)) #intensityを修正
    # fig1 = plt.figure()
    # fig1.set_figwidth(25)
    # fig1.set_figheight(25)
    # ax1 = fig1.add_subplot(2, 1, 1)
    # ax2 = fig1.add_subplot(2, 1, 2)
    # plt.title(str(time[i])+"~"+str(time[i+step]))
    # ax1.plot(time[i:i+step], ecg_pulse[i:i+step])
    # ax1.set_xlabel("Time[s]")
    # ax1.set_ylabel("Vol[v]")
    # ax2.plot(freq, Amp)
    # ax2.set_xlim(0,50)
    # ax2.set_xlabel("Frequency(Hz)")
    # ax2.set_ylabel("Amplitude")
    # plt.show()
    
    freq, ts, Zxx = signal.stft(ecg_pulse[i:i+step], fs, nperseg=1000, noverlap=850)
    k = 0
    # T_Zxx = Zxx.T
    # T_Zxx = min_max_n(T_Zxx)
    Zxx_abs = np.abs(Zxx)
    Zxx_df = pd.DataFrame(Zxx_abs)
    Zxx_df = min_max_df(Zxx_df)
    # while(k!= len(ts)):
    #     min_max_n(Zxx_df, k)
    #     k += 1
    #     time.sleep(0.0001)
    
    Zxx = Zxx_df.values
    fig2 = plt.figure()
    fig2.set_figwidth(25)
    fig2.set_figheight(25)
    ax3 = fig2.add_subplot(2, 1, 1)
    ax4 = fig2.add_subplot(2, 1, 2)
    plt.title(str(t[i])+"~"+str(t[i+step]))
    ax3.plot(t[i:i+step], ecg_pulse[i:i+step])
    ax3.set_xlabel("Time[s]")
    ax3.set_ylabel("Vol[v]")
    ax4.pcolormesh(ts, freq, np.abs(Zxx)) #intensityを修正
    ax4.set_ylim(0, 30)
    # cbar = fig2.colorbar()
    ax4.set_xlabel("Time(sec)")
    ax4.set_ylabel("Frequency(Hz)")
    plt.pcolormesh(ts, freq, np.abs(Zxx)) #intensityを修正
    plt.ylim(0, 10)
    plt.colorbar(extendfrac=(0, 1.0))
    plt.xlabel("Time(sec)")
    plt.ylabel("Frequency(Hz)")
    # plt.show()
    fig2.savefig('img/stfft/stfft_'+ str(t[i])+"~"+str(t[i+step])+'.png')