from scipy import signal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fft_wave
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import time
import peek_pridict
import pickle
import auto_thread

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

def create_windowdata(ecg_pulse, ecg_heart, num): #脈波系データ, 心電図データ, 窓幅
    train_pulse = np.array(ecg_pulse)
    train_heart = np.array(ecg_heart)
    pulse_X = np.empty((len(ecg_pulse),num))
    
    for i in range(num):
        pulse_X[:, i] = np.roll(train_pulse, -i)
        
    pulse_X = pulse_X[num-1: -num+1]
    train_pulse = train_pulse[num-1 : -num+1]
    train_heart = train_heart[num-1 : -num+1]
    
    print(pulse_X.shape)
    print(train_pulse.shape)
        
    return pulse_X, train_heart


if __name__ == "__main__":
    #学習済みデータの読み込み
    model = pickle.load(open('wave_study.sav', 'rb'))
    
    #初期値定義
    before_start = time.time()
    n = int(input('ファイル番号を指定してください:'))
    fs = 500
    t = np.arange(0, 600, 1/ fs)
    # ecg_heart_low, ecg_pulse = fft_wave.read_data(n, fs, 1) #silver
    ecg_heart_low, ecg_pulse = fft_wave.read_data(n, fs, 2) #file
    ecg_pulse = fft_wave.lowpass(ecg_pulse, 5, 20, fs)
    ecg_heart_low = fft_wave.lowpass(ecg_heart_low, 5, 20, fs)
    ecg_heart = fft_wave.wave_diff(ecg_heart_low) #波形データ
    num = 100 #窓幅
    #サンプルデータ
    train_pulse, train_heart = create_windowdata(ecg_pulse[0: 3000000], ecg_heart, num) #検証用脈波、検証用心電
    
    #検証
    pred = model.predict(train_pulse)
    
    fig = plt.figure()
    fig.set_figwidth(20)
    # plt.plot(ecg_heart_low[0: 5000], label="heart_20-filter")
    plt.plot(ecg_heart_low[0: 300000], label="heart")
    plt.plot(ecg_pulse[0: 300000], label="pulse")
    plt.plot(pred[0: 300000], label="pred")
    plt.legend()
    plt.show()
    
    th_heart = auto_thread.auto_thread(ecg_heart)
    # time_heart = peek_pridict.get_peaks(ecg_heart, th_heart, 0, 300000)
    time_heart = peek_pridict.get_peaks_lower(ecg_heart, th_heart, 0, 300000)
    time_heart = np.array(time_heart)
    th_rpulse = auto_thread.auto_thread(ecg_pulse)
    time_rpulse = peek_pridict.get_peaks(ecg_pulse, th_rpulse, 0, 300000)
    time_rpulse = np.array(time_rpulse)
    # th_ppulse = auto_thread.auto_thread(pred)
    th_ppulse = 0.01
    time_ppulse = peek_pridict.get_peaks(pred, th_ppulse, 0, 300000)
    time_ppulse = np.array(time_ppulse)
    
    #ピーク値が取得できていない場合の対処
    heart_ave = np.average(time_heart)
    for i in range(1, len(time_ppulse)):
        if time_ppulse[i] - time_ppulse[i-1] > heart_ave:
            np.insert(time_ppulse, )
    
    databox = pd.DataFrame(index=[], 
                           columns=['心拍時', '脈拍時', '疑似拍時', '心脈時差', '心予時差', '心拍間隔',
                                    '脈拍間隔', '疑似心拍間隔', '心拍-脈拍', '心拍-疑似'])
    
    print(len(time_heart))
    print(len(time_rpulse))
    print(len(time_ppulse))
    
    if len(time_rpulse) < len(time_heart):
        for i in range(len(time_rpulse), len(time_heart)):
            time_rpulse = np.append(time_rpulse, np.nan)
    elif len(time_rpulse) > len(time_heart):
        for i in range(len(time_heart), len(time_rpulse)):
            time_heart = np.append(time_heart, np.nan)
    
    if len(time_ppulse) < len(time_heart):
        for i in range(len(time_ppulse), len(time_heart)):
            time_ppulse = np.append(time_ppulse, np.nan)
    elif len(time_ppulse) > len(time_heart):
        for i in range(len(time_heart), len(time_ppulse)):
            time_heart = np.append(time_heart, np.nan)
            
    if len(time_ppulse) < len(time_rpulse):
        for i in range(len(time_ppulse), len(time_rpulse)):
            time_ppulse = np.append(time_ppulse, np.nan)
    elif len(time_ppulse) > len(time_rpulse):
        for i in range(len(time_rpulse), len(time_ppulse)):
            time_rpulse = np.append(time_rpulse, np.nan)
    
    databox = databox.assign(心拍時=time_heart)
    databox = databox.assign(脈拍時=time_rpulse)
    databox = databox.assign(疑似拍時=time_ppulse)
    databox.to_csv('data/timedata_pd.csv', encoding='utf_8_sig')