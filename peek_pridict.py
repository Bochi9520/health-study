import numpy as np
import pandas as pd
import fft_wave
import matplotlib.pyplot as plt
from scipy import signal
    
def get_peaks(data, threshold, range1, range2):
    """ 
    データの変化量を順にみて、
    極大 (変化量が + から - に変わる地点)、極小 (- から +) のインデックスをそれぞれ取得 
    """

    if data.size < 2 :
        raise Exception("Data size is too few.")

    # 変化量算出
    diffs = np.zeros((data.size - 1))
    for i in range(data.size - 1) :
        idx = i + 1
        diffs[i] = data[idx] - data[idx - 1]
        
    # 符号
    def get_flag(diff) :
        if diff > 0 : return 1
        if diff < 0 : return -1
        return 0

    peak_max = []
    maxid = []

    pre_flag = get_flag(diffs[0])

    for i in range(1, diffs[0:300000].size, 300):
        maxid = 0
        max = np.amin(data) #最小値

        for j in range(i, i+300):
            flag = get_flag(diffs[j])

            # 今の変化量が 0 なら無視
            if flag == 0 : continue

            # 前の変化量が 0 or 今の変化量が 0 以外 → 設定
            # NOTE: 0 番目の変化量から 0 で続いていたときのみ
            if pre_flag == 0 :
                if flag <= 0 and max < data[j] and data[j] > threshold: 
                    # peak_max.append(j)
                    maxid = j
                    max = data[j]
                pre_flag = flag
                continue

            # 変化量が前と違う場合は設定
            if pre_flag != flag :
                if flag <= 0 and max < data[j] and data[j] > threshold:
                    # peak_max.append(j)
                    maxid = j
                    max = data[j]
                pre_flag = flag
        
        peak_max.append(maxid)
            
            
    peak_max = np.array(peak_max)
    peak_max = peak_max[peak_max != 0]
    i = 1
    
    while i < len(peak_max):
        if i == len(peak_max):
            break
        elif peak_max[i] - peak_max[i-1] < 250 and data[peak_max[i]] < data[peak_max[i-1]]:
            peak_max = np.delete(peak_max, i)

        elif peak_max[i] - peak_max[i-1] < 250 and data[peak_max[i]] > data[peak_max[i-1]]:
            peak_max = np.delete(peak_max, i-1)
        
        else:
            i += 1
            
    t = np.arange(0, 600, 1/500)
    fig1 = plt.figure()
    fig1.set_figwidth(20)
    plt.plot(t[range1:range2], data[range1:range2])
    plt.plot(t[peak_max[range1:range2]], data[peak_max[range1:range2]], "ro")
    plt.show()
    
    return peak_max[range1:range2]

def get_peaks_lower(data, threshold, range1, range2):
    """ 
    データの変化量を順にみて、
    極大 (変化量が + から - に変わる地点)、極小 (- から +) のインデックスをそれぞれ取得 
    """

    if data.size < 2 :
        raise Exception("Data size is too few.")

    # 変化量算出
    diffs = np.zeros((data.size - 1))
    for i in range(data.size - 1) :
        idx = i + 1
        diffs[i] = data[idx] - data[idx - 1]
        
    # 符号
    def get_flag(diff) :
        if diff > 0 : return 1
        if diff < 0 : return -1
        return 0

    peak_min = []
    minid = []

    pre_flag = get_flag(diffs[0])

    for i in range(1, diffs[0:300000].size, 300):
        minid = 0
        min = np.amax(data) #最大値

        for j in range(i, i+300):
            flag = get_flag(diffs[j])

            # 今の変化量が 0 なら無視
            if flag == 0 : continue

            # 前の変化量が 0 or 今の変化量が 0 以外 → 設定
            # NOTE: 0 番目の変化量から 0 で続いていたときのみ
            if pre_flag == 0 :
                if flag >= 0 and min > data[j] and data[j] < threshold: 
                    # peak_max.append(j)
                    minid = j
                    min = data[j]
                pre_flag = flag
                continue

            # 変化量が前と違う場合は設定
            if pre_flag != flag :
                if flag >= 0 and min > data[j] and data[j] < threshold:
                    # peak_max.append(j)
                    minid = j
                    min = data[j]
                pre_flag = flag
        
        peak_min.append(minid)
            
            
    peak_min = np.array(peak_min)
    peak_min = peak_min[peak_min != 0]
    i = 1
    
    while i < len(peak_min):
        if i == len(peak_min):
            break
        elif peak_min[i] - peak_min[i-1] < 250 and data[peak_min[i]] > data[peak_min[i-1]]:
            peak_min = np.delete(peak_min, i)

        elif peak_min[i] - peak_min[i-1] < 250 and data[peak_min[i]] < data[peak_min[i-1]]:
            peak_min = np.delete(peak_min, i-1)
        
        else:
            i += 1
            
    t = np.arange(0, 600, 1/500)
    fig1 = plt.figure()
    fig1.set_figwidth(20)
    plt.plot(t[range1:range2], data[range1:range2])
    plt.plot(t[peak_min[range1:range2]], data[peak_min[range1:range2]], "ro")
    plt.show()
    
    return peak_min[range1:range2]

def get_peaks_pulse(data, threshold, range1, range2):
    """ 
    データの変化量を順にみて、
    極大 (変化量が + から - に変わる地点)、極小 (- から +) のインデックスをそれぞれ取得 
    """

    if data.size < 2 :
        raise Exception("Data size is too few.")

    # 変化量算出
    diffs = np.zeros((data.size - 1))
    for i in range(data.size - 1) :
        idx = i + 1
        diffs[i] = data[idx] - data[idx - 1]
        
    # 符号
    def get_flag(diff) :
        if diff > 0 : return 1
        if diff < 0 : return -1
        return 0

    peak_max = []
    # max = data[0]

    pre_flag = get_flag(diffs[0])

    # for i in range(1, diffs[0:299701].size, 300):
    # for i in range(1, diffs[0:300000].size, 300):
    for i in range(1, diffs.size-300, 300):
        maxid = 0
        max = np.amin(data)
        for j in range(i, i+300):
            flag = get_flag(diffs[j])
            # 今の変化量が 0 なら無視
            if flag == 0 : continue

            # 前の変化量が 0 or 今の変化量が 0 以外 → 設定
            # NOTE: 0 番目の変化量から 0 で続いていたときのみ
            if pre_flag == 0 :
                if flag <= 0 and max < data[j] and data[j] > threshold: 
                    # peak_max.append(j)
                    maxid = j
                    max = data[j]
                pre_flag = flag
                continue

            # 変化量が前と違う場合は設定
            if pre_flag != flag :
                if flag <= 0 and max < data[j] and data[j] > threshold:
                    # peak_max.append(j)
                    maxid = j
                    max = data[j]
                pre_flag = flag
        
        peak_max.append(maxid)
        
    peak_max = np.array(peak_max)
    peak_max = peak_max[peak_max != 0]
    i = 1
    while i < len(peak_max):
        if i == len(peak_max):
            break
        elif peak_max[i] - peak_max[i-1] < 250 and data[peak_max[i]] < data[peak_max[i-1]]:
            peak_max = np.delete(peak_max, i)
        
        elif peak_max[i] - peak_max[i-1] < 250 and data[peak_max[i]] > data[peak_max[i-1]]:
            peak_max = np.delete(peak_max, i-1)
    
        else:
            i += 1
    
    t = np.arange(0, 600, 1/500)
    fig1 = plt.figure()
    fig1.set_figwidth(20)
    plt.plot(t[range1:range2], data[range1:range2])
    plt.plot(t[peak_max[range1:range2]], data[peak_max[range1:range2]], "ro")
    plt.show()

    return peak_max[range1:range2]
               

if __name__ == "__main__":
    heart, pulse, elbor = fft_wave.read_data(19)
    pulse = fft_wave.lowpass(pulse, 5, 20, 500)
    elbor = fft_wave.lowpass(elbor, 5, 20, 500)
    heart = np.array(heart)
    pulse = np.array(pulse)
    elbor = np.array(elbor)
    t = np.arange(0, 600, 1/500)
    th_pulse = float(input("閾値を設定してください:"))
    th_elbor = float(input("閾値(elbor)を設定してください:"))
    # peek_max = np.array(get_peaks(heart, th, 0, 300000))
    peek_pulse = get_peaks_pulse(pulse[0: 300000], th_pulse, 0, 300000)
    peek_pulse = np.array(peek_pulse)
    peek_elbor = get_peaks_pulse(elbor[0: 300000], th_elbor, 0, 300000)
    peek_elbor = np.array(peek_elbor)
    peek_pulse = peek_pulse[peek_pulse != 0]
    peek_elbor = peek_elbor[peek_elbor != 0]
    time_diff = np.empty(0)
    # maxid = signal.argrelmax(pulse[0: 300000], order=200)
    fig = plt.figure()
    fig.set_figwidth(20)
    # plt.plot(t, heart[0: 300000], "b", alpha=0.5)
    # plt.plot(t, pulse[0: 300000], "b", alpha=0.5)
    # plt.plot(t[229000: 252500], pulse[229000: 252500], "b", alpha=0.5)
    # plt.plot(t[0: 300000], pulse[0: 300000], "b", alpha=0.5)
    plt.plot(t[0: 300000], elbor[0: 300000], "b", alpha=0.5)
    # plt.plot(t[peek_max], heart[peek_max], 'ro')
    # plt.plot(t[peek_pulse], pulse[peek_pulse], 'ro')
    plt.plot(t[peek_elbor], elbor[peek_elbor], 'ro')
    plt.show()
    
    print(len(peek_pulse))
    print(len(peek_elbor))
    
    # for i in range(0, len(peek_max)-1):
    #     if peek_max[i] >= 240000:
    #         time_diff = np.append(time_diff, peek_max[i+1] - peek_max[i])
    
    for i in range(0, len(peek_pulse)):
        if 150000 <= peek_pulse[i] <=200000:
            time_diff = np.append(time_diff, peek_pulse[i] - peek_elbor[i+2])
    
    # time_diff = np.append(time_diff, 0)
    # databox = pd.read_csv('data/timedata_pd.csv')
    # databox = databox.assign(脈拍間隔=time_diff)
    # databox.to_csv('data/timedata_pd.csv', encoding='utf_8_sig')
    
    #肘と手首の時差記録
    databox = pd.DataFrame(index=[], 
                           columns=['脈拍時差'])
    databox = databox.assign(脈拍時差=time_diff)
    databox.to_csv('data/diffdata_pd.csv', encoding='utf_8_sig')