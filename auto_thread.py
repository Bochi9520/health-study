import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fft_wave
import peek_pridict

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
    "safe_ito_20230801.csv" #20
]

def auto_thread(data):
    data = np.array(data)
    # ラベルの初期化
    labels = np.random.randint(0,2,data.shape[0])

    # 終了条件
    OPTIMIZE_EPSILON = 1

    m_0_old = -np.inf
    m_1_old = np.inf

    for i in range(1000):
    # それぞれの平均の計算
        m_0 = data[labels==0].mean()
        m_1 = data[labels==1].mean()
    # ラベルの再計算
        labels[np.abs(data-m_0) < np.abs(data-m_1)] = 0
        labels[np.abs(data-m_0) >= np.abs(data-m_1)] = 1
    #     終了条件
        if np.abs(m_0 - m_0_old) + np.abs(m_1 - m_1_old) < OPTIMIZE_EPSILON:
            break
        m_0_old = m_0
        m_1_old = m_1
    # 初期値によって，クラスが変化するため上界の小さい方を採用
    thresh_kmeans = np.minimum(data[labels==0].max(),data[labels==1].max())
    
    return thresh_kmeans

if __name__ == "__main__":
    # データの読み込み
    fs = 500
    n = int(input('ファイル番号を指定してください:'))
    ecg_heart, ecg_pulse = fft_wave.read_data(n, fs)
    ecg_heart = fft_wave.lowpass(ecg_pulse, 5, 20, 500)
    ecg_pulse = fft_wave.lowpass(ecg_pulse, 5, 20, 500)
    
    data = np.array(ecg_heart)
    th = auto_thread(data)
    print(th)
    peek_pridict.get_peaks(data, th, 0, 300000)