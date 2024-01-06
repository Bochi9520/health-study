import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fft_wave
import auto_thread
import peek_pridict

def derivative(data, i, h):
    return (data(i + h) - data(i - h)) / (2 * h)

#ピーク検出関数(波形, 閾値, 開始時刻, 終点時刻, 窓間隔, オーバーラップ)
def new_get_peaks(data, threshold, range1, range2, window, overrap):
    """ 
    データ窓ごとにデータの変化量を順にみて、
    極大 (変化量が + から - に変わる地点)、極小 (- から +) のインデックスをそれぞれ取得 
    """
    
    if data.size < 2 :
        raise Exception("Data size is too few.")
    
    #変化量算出
    diffs = np.zeros((data.size - 1))
    for i in range(data.size - 1):
        idx = i + 1
        diffs[i] = data[idx] - data[idx - 1]  
        
    #符号付与
    def get_flag(diff):
        if diff > 0: 
            return 1
        if diff < 0: 
            return -1
        else :return 0
    
    peak_max = []
    
    #ピーク判別
    pre_flag = get_flag(diffs[0]) #開始地点設定
    for i in range(1, diffs.size - window, overrap):
        peakid = 0 #ピーク地点の記録変数
        max = data.min() #最小値を最大値に設定(走査窓ごとで再設定)
        for j in range(i, i+window):
            flag = get_flag(diffs[j])
            # 今の変化量が 0 なら無視
            if flag == 0 : continue
            
            #前の変化量(preflag)が1 and 今の変化量が0 => 設定
            if pre_flag == 0:
                if flag == -1 and max < data[j] and data[j] > threshold:
                    max = data[j] #走査窓内で一点だけを求めるため
                    peakid = j
                pre_flag = flag     
                continue
                   
            #前の変化量(preflag)が1 and 今の変化量が-1 => 設定
            elif pre_flag != flag:
                if flag == -1 and max < data[j] and data[j] > threshold:
                    max = data[j] #走査窓内で一点だけを求めるため
                    peakid = j
                pre_flag = flag
            
        peak_max.append(peakid)
        
    peak_max = np.array(peak_max)
    peak_max = peak_max[peak_max != 0]
    i = 1
    
    while i < len(peak_max):
        if peak_max[i] == peak_max[i-1]:
            peak_max = np.delete(peak_max, i)        
        else:
            i += 1
            
    i = 1
    while i < len(peak_max):
        if peak_max[i] - peak_max[i-1] < 250:
            peak_max = np.delete(peak_max, i)
        else:
            i += 1
    
    return peak_max[range1: range2]

#ピーク検出関数(心拍間隔, 間隔平均, 心電図)
def check_arrhythmia(rri, ave_rri, heartpeak, diffheartpeak):
    for i in range(1, len(rri)-1):
        if (rri[i-1] < ave_rri-50 or rri[i-1] > ave_rri+50) and (ave_rri*2-50 <= (rri[i-1]+rri[i+1]) <= ave_rri*2+50):
            print('心房性不整脈検知')
        elif(len(diffheart_peak) - len(heartpeak) >= 10):
            print('心室性不整脈検知')
        else:
            print('正常脈です')
            


if __name__ == "__main__":
    #入力値設定
    kind = input('ファイル系等を指定して下さい: ') #ファイル系等 sil: シルバー, st: 学生
    n = int(input('使用するデータを指定してください: '))
    fs = 500 #サンプリング周波数
    t = np.arange(0, 600, 1/fs)
    window = 300 #窓数
    window = 50 #オーバーラップ
    
    #波形情報抽出
    if kind == 'sil':
        heart, pulse, elbor = fft_wave.read_data_silver(n, fs) #silver
    else:
        heart, pulse, elbor = fft_wave.read_data(n, fs) #file
    heart = fft_wave.lowpass(heart[0:len(t)], 5, 20, 500)
    pulse = fft_wave.lowpass(pulse, 5, 20, 500)
    elbor = fft_wave.lowpass(elbor, 5, 20, 500)
    diff_heart = np.gradient(heart)
    diff_pulse = np.gradient(pulse)
    
    #閾値設定
    # th_heart = auto_thread.auto_thread(heart)
    th_heart = heart.max() * 0.8
    # th_diffheart = auto_thread.auto_thread(diff_heart)
    th_diffheart = diff_heart.max() *0.6
    # th_pulse = auto_thread.auto_thread(pulse)
    th_pulse = pulse.max() * 0.8
    # th_diffheart = auto_thread.auto_thread(diff_heart)
    th_diffpulse = diff_pulse.max() *0.6
    th_elbor = auto_thread.auto_thread(elbor)

    #ピーク検出関数(波形, 閾値, 開始時刻, 終点時刻, 窓間隔, オーバーラップ)
    heart_peak = new_get_peaks(heart, th_heart, 0, len(heart), 300, 25)
    print('生心電図のピーク数:' + str(len(heart_peak)))
    diffheart_peak = new_get_peaks(diff_heart, th_diffheart, 0, len(diff_heart), 300, 25)
    print('微分心電図のピーク数:' + str(len(diffheart_peak)))
    
    #ピーク検出関数_脈波(波形, 閾値, 開始時刻, 終点時刻, 窓間隔, オーバーラップ)
    pulse_peak = new_get_peaks(pulse, th_pulse, 0, len(pulse), 300, 25)
    print('生脈波のピーク数:' + str(len(pulse_peak)))
    diffpulse_peak = new_get_peaks(diff_pulse, th_diffpulse, 0, len(diff_pulse), 300, 25)
    print('微分脈波のピーク数:' + str(len(diffpulse_peak)))
    
    #旧ピーク検出関数(波形, 閾値, 開始時刻, 終点時刻, 窓間隔, オーバーラップ)
    old_heart_peak = peek_pridict.get_peaks(heart, th_heart, 0, len(heart))
    print('生心電図のピーク数(旧):' + str(len(old_heart_peak)))
    old_diffheart_peak = peek_pridict.get_peaks(diff_heart, th_diffheart, 0, len(diff_heart))
    print('微分心電図のピーク数(旧):' + str(len(old_diffheart_peak)))
    
    #旧ピーク検出関数_脈波(波形, 閾値, 開始時刻, 終点時刻, 窓間隔, オーバーラップ)
    old_pulse_peak = peek_pridict.get_peaks(pulse, th_pulse, 0, len(heart))
    print('生脈波のピーク数(旧):' + str(len(old_pulse_peak)))
    old_diffpulse_peak = peek_pridict.get_peaks(diff_pulse, th_diffpulse, 0, len(diff_heart))
    print('微分脈波のピーク数(旧):' + str(len(old_diffpulse_peak)))
    
    #心拍間隔の算出
    diffheart_peaktime = []
    RRI = []
    for i in range(len(diffheart_peak)):
        diffheart_peaktime.append(diffheart_peak[i]/500)
    for i in range(1, len(diffheart_peak)):
        RRI.append(diffheart_peak[i] - diffheart_peak[i -1])
    
    # print(diffheart_peaktime)
    print('心拍間隔のリスト', end=':')
    print(RRI)
    print('心拍間隔の平均', end=':')
    print(int(np.average(RRI)))
    
    RRI_diff = []
    for i in range(1, len(RRI)):
        RRI_diff.append(RRI[i] - RRI[i-1])
        
    print('心拍間隔の差のリスト', end=':')
    print(RRI_diff)
    
    
    
    
    #図の描画
    # fig = plt.figure()
    # fig.set_figwidth(20)
    # fig.set_figheight(40)
    # ax1 = fig.add_subplot(4, 1, 1)
    # ax2 = fig.add_subplot(4, 1, 2)
    # ax3 = fig.add_subplot(4, 1, 3)
    # ax4 = fig.add_subplot(4, 1, 4)
    # ax1.plot(t[0: len(heart)], pulse[0: 300000], "b", alpha=0.5)
    # ax1.plot(t[pulse_peak], pulse[pulse_peak], 'ro')
    # ax2.plot(t[0: len(pulse)], pulse[0: 300000], "b", alpha=0.5)
    # ax2.plot(t[old_pulse_peak], pulse[old_pulse_peak], 'ro')
    # ax3.plot(t[0: len(heart)], diff_heart[0: 300000], "b", alpha=0.5)
    # ax3.plot(t[diffheart_peak], diff_heart[diffheart_peak], "ro")
    # ax4.plot(t[0: len(diff_heart)], diff_heart[0: 300000], "b", alpha=0.5)
    # ax4.plot(t[old_diffheart_peak], diff_heart[old_diffheart_peak], "ro")
    # plt.show()


    