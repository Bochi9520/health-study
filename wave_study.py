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

def pulsepeek_piucture(y,lab,ID):  #脈波系データ, 心電図データ, 画像ラベル
    fig = plt.figure()
    fig.set_figwidth(20)
    t = np.arange(0, len(y))
    plt.plot(t, y, label=lab, c="blue", alpha=0.5)
    plt.plot(t[ID], y[ID], "ro")
    plt.xlabel("time")
    plt.ylabel("ECG")
    plt.legend(loc="upper right")
    plt.show()
    
    peek_time = t[ID]
    return peek_time
    
def heartpeek_piucture(ecg):  #脈波系データ, 心電図データ, 画像ラベル
    fig = plt.figure()
    fig.set_figwidth(20)
    t = np.arange(0, 600, 1/500)
    maxid = signal.argrelmax(ecg[:len(t)], order = 250)
    plt.plot(t, ecg[:len(t)], c="blue", alpha=0.5)
    plt.plot(t[maxid], ecg[maxid], "ro")
    plt.xlabel("time")
    plt.ylabel("PULSE")
    plt.show()
    peek_time = int(t[maxid])
    return peek_time
    

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
        
    return train_pulse, train_heart, pulse_X

def learning(train_pulse, train_heart, pulse_data, test_data, test_pulse):  #教師用脈波形, 教師用心電図, 学習用脈波パルス, 検証用別データ
    x_train= pulse_data[:240000]
    # y_train = train_pulse[:240000] #脈波教師
    y_train= train_heart[:240000] #心電教師
    x_test = test_data[239750: 300000]
    y_test = test_pulse[239750: 300000]
    clf = MLPRegressor(hidden_layer_sizes=(100, 20), max_iter= 200, random_state=1, verbose=True, 
                       early_stopping=True, learning_rate="adaptive")
    clf.fit(x_train, y_train)
    print(clf.score(x_test, y_test))
    y_pred = clf.predict(x_test)
    print(clf.get_params(deep=True))
    filename = 'wave_study.sav'
    pickle.dump(clf, open(filename, 'wb'))
    pd.DataFrame(clf.loss_curve_).plot()
    pd.DataFrame(clf.validation_scores_).plot()
    
    return y_test, y_pred

def deleat_tpulse(wave, pretime):
    heart_pulse = wave
    for i in range(0, pretime[0]-5):
        heart_pulse[i] = 0
        
    for i in range(1, len(pretime)):
        for j in range(pretime[i-1]+5, pretime[i]-5):
            heart_pulse[j] = 0
            
    for i in range(pretime[-1]+5, len(wave)):
        heart_pulse[i] = 0
        
    return heart_pulse
    
if __name__ == "__main__":
    #初期値定義
    before_start = time.time()
    fs = 500
    n = int(input('学習用ファイル番号を指定してください:'))
    m = int(input('test用ファイル番号を指定してください:'))
    t = np.arange(0, 600, 1/ 500)
    ecg_heart, ecg_pulse = fft_wave.read_data(n, fs)
    ecg_heart = fft_wave.lowpass(ecg_heart, 5, 20, 500)
    ecg_pulse = fft_wave.lowpass(ecg_pulse, 5, 20, 500)
    test_h, test_p = fft_wave.read_data(m, fs)
    num = 300 #窓幅
    
    #パルス波生成
    th_heart = auto_thread.auto_thread(ecg_heart)
    pulse_heart = np.array(ecg_heart[0: 3000000])
    time_heart = peek_pridict.get_peaks(pulse_heart, th_heart, 0, 300000)
    pulse_heart = deleat_tpulse(pulse_heart, time_heart)
    
    fig1 = plt.figure()
    fig1.set_figwidth(20)
    plt.plot(t[280000:285000], pulse_heart[280000:285000])
    plt.ylabel('電圧[V]', fontname="MS Gothic")
    plt.xlabel('時間[s]', fontname="MS Gothic")
    plt.show()
    
    # th_heart = float(input("閾値を設定してください(心電):")) #2.07
    th_heart = auto_thread.auto_thread(test_h)
    # th_pulse = float(input("閾値を設定してください(予測波形):"))
    th_pulse = auto_thread.auto_thread(test_p)
    test_h = fft_wave.lowpass(test_h, 5, 20, 500)
    test_h_diff = fft_wave.wave_diff(test_h) #波形データ
    # test_h = np.array(test_h)
    test_p = fft_wave.lowpass(test_p, 5, 20, 500)
    test_p = np.array(test_p)
    # time_heart = peek_pridict.get_peaks(test_h, th_heart, 0, 300000)
    time_heart = peek_pridict.get_peaks_lower(test_h_diff, th_heart, 0, 300000)
    time_heart = np.array(time_heart)
    time_rpulse = peek_pridict.get_peaks(test_p, th_pulse, 0, 300000)
    time_rpulse = np.array(time_rpulse)
    
    
    #学習適用データ作成
    # train_pulse, train_heart, pulse_date = create_windowdata(ecg_pulse, ecg_heart, num) #教師用脈波、教師用心電、学習用心電
    train_pulse, train_heart, pulse_date = create_windowdata(ecg_pulse[0: 3000000], pulse_heart, num) #教師用脈波、教師用心電、学習用心電
    test_pulse, test_heart, test_data = create_windowdata(test_p[0: 3000000], test_h[0: 3000000], num) #検証用脈波、検証用心電、検証用窓変換脈波

    befoer_end = time.time()
    print("前処理の処理時間:"+ str(befoer_end - before_start))
    learning_start = time.time()
    real_pulse, pred_pulse = learning(train_pulse, train_heart, pulse_date, test_data, test_pulse)
    learning_end = time.time()
    print("学習時間:"+ str(learning_end - learning_start))
    
    
    fig = plt.figure()
    fig.set_figwidth(20)
    fig.set_figheight(40)
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2)
    ax3 = fig.add_subplot(3, 1, 3)
    ax1.plot(t[0: 10000],test_h[240000: 250000], color="red", label="heart")
    ax2.plot(t[0: 10000],test_p[240000: 250000], label="pulse")
    ax3.plot(t[0: 10000],pred_pulse[0: 10000], color= "green")
    plt.legend()
    plt.show()
    fig.savefig("data/compare.png")
    
    
    maxid = signal.argrelmax(pred_pulse, order=350)
    time_pulse = pulsepeek_piucture(pred_pulse, '予測波形', maxid)
    
    databox = pd.DataFrame(index=[], 
                           columns=['心拍時', '脈拍時', '疑似拍時', '心脈時差', '心予時差', '心拍間隔',
                                    '脈拍間隔', '疑似心拍間隔', '心拍-脈拍', '心拍-疑似'])

    j = 0
    time_array_heart = np.empty(0)
    for i in range(0, len(time_heart)):
        if time_heart[i] >= 240000:
            time_array_heart = np.append(time_array_heart, (time_heart[i]-240000))
            j += 1
    
    time_array_rpulse = np.empty(0)
    
    j = 0
    for i in range(0, len(time_rpulse)):
        if time_rpulse[i] >= 240000:
            time_array_rpulse = np.append(time_array_rpulse, (time_rpulse[i]-240000))
            j += 1
            
    if len(time_array_rpulse) < len(time_array_heart):
        for i in range(len(time_array_rpulse), len(time_array_heart)):
            time_array_rpulse = np.append(time_array_rpulse, np.nan)
    elif len(time_array_rpulse) > len(time_array_heart):
        for i in range(len(time_array_heart), len(time_array_rpulse)):
            time_array_heart = np.append(time_array_heart, np.nan)

        
    time_array_pulse = np.empty(0)
    for i in range(0, len(time_pulse)):
        time_array_pulse = np.append(time_array_pulse, time_pulse[i])
    
    if len(time_array_pulse) < len(time_array_heart):
        for i in range(len(time_array_pulse), len(time_array_heart)):
            time_array_pulse = np.append(time_array_pulse, np.nan)
    elif len(time_array_pulse) > len(time_array_heart):
        for i in range(len(time_array_heart), len(time_array_pulse)):
            time_array_heart = np.append(time_array_heart, np.nan)

    if len(time_array_rpulse) < len(time_array_pulse):
        for i in range(len(time_array_rpulse), len(time_array_pulse)):
            time_array_rpulse = np.append(time_array_rpulse, np.nan)
    elif len(time_array_rpulse) > len(time_array_pulse):
        for i in range(len(time_array_pulse), len(time_array_rpulse)):
            time_array_pulse = np.append(time_array_pulse, np.nan)

    # time_array = np.append(time_array, 0)
    print(len(time_array_heart))
    print(len(time_array_rpulse))
    print(len(time_array_pulse))
    databox = databox.assign(心拍時=time_array_heart)
    databox = databox.assign(脈拍時=time_array_rpulse)
    databox = databox.assign(疑似拍時=time_array_pulse)
    databox.to_csv('data/timedata_pd.csv', encoding='utf_8_sig')
    
    
   