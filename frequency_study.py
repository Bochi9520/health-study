import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from scipy import signal
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report, jaccard_score
import fft_wave
import peek_pridict
import pickle
import auto_thread
import check_study as cs

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

def create_dataset(wave, fs):
    pulse = fft_wave.lowpass(wave, 5, 30, 500)
    pulse = fft_wave.highpass(pulse, 2, 0.5, fs)
    freq, ts, Zxx = signal.stft(pulse, fs, nperseg = 1000, noverlap = 950)
    phase_data = np.angle(Zxx)
    Zxx_abs = np.abs(Zxx)
    pd_freq = pd.DataFrame(Zxx_abs, columns=ts, index=freq)
    print(pd_freq.head())
    pd_phase = pd.DataFrame(phase_data, columns= pd_freq.columns, index= pd_freq.index) 
    #各周波数帯の分散値 axis 0:列, 1:行
    # pd_freq_var = pd_freq[0:25].var(axis=1)
    # print(pd_freq_var)    
    
    #各周波数帯の四分位範囲 first:第一四分位, third:第三四分位
    freq_quant_first = pd_freq.quantile(0.25, axis=0)
    q3_values = pd_freq.quantile(0.75, axis=0)
    
    # 列ごとに最大値を取得し、その行の名前を取得
    max_row_names = pd_freq.idxmax()
    
    # 列ごとに二番目に大きい値を取得
    second_max_values = pd_freq.apply(lambda col: col.nlargest(2).iloc[-1])

    # 結果を新しいDataFrameに格納
    result_df = pd.DataFrame({'Max_Row_Name': max_row_names, 'Second_Max_Value': second_max_values, 'Q3': q3_values})

    # 結果をExcelファイルに保存
    result_df.to_csv('data/max_row_names.csv', index=True)
    
    #係数dataframe
    # 行数と列数を定義
    rows = 501

    # 初期のDataFrameを生成
    df = pd.DataFrame(0, columns=ts, index=freq)

    # 2行目から6行目までの各行を1に設定
    for i in range(1, 12):
        df.iloc[i, :] = 1

    # 残りの行を1/3倍して設定
    power = 1
    for i in range(12, rows):
        df.iloc[i, :] = df.iloc[i - 3, :] / 6**power
        if (i - 5) % 3 == 0:
            power += 1

    # DataFrameを表示（表示が長い場合は必要な部分だけ表示できます）
    repd_freq = pd_freq*df
    re_zxx = repd_freq * np.exp(1j * pd_phase)
    # re_zxx = re_zxx.values
    # re_zxx = pd_freq.values
    re_t, re_pulse = signal.istft(re_zxx, fs, nperseg = 1000, noverlap = 950)
    fig = plt.figure()
    fig.set_figwidth(20)
    fig.set_figheight(40)
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2)
    ax3 = fig.add_subplot(3, 1, 3)
    # ax1.plot(re_t[:6001], pulse[:175001])
    ax1.plot(re_t[:175001], pulse[:175001])
    ax2.plot(re_t[:175001], re_pulse[:175001])
    re_pulse = fft_wave.lowpass(re_pulse, 5, 4, 500)
    ax3.plot(re_t[:175001], re_pulse[:175001])
    plt.show()
    
    return re_pulse

def create_label(data):
    # データの長さ
    data_length = len(repulse)
    # セグメント長とオーバーラップ
    segment_length = 30
    overlap = 25
    # セグメント数を計算
    num_segments = (data_length - segment_length) // (segment_length - overlap)
    # 二次元データの初期化
    x_data = np.zeros((num_segments, segment_length)) 
    y_data = np.zeros((num_segments, 1)) 
    lavel_cunt = 1
    true = [1] #脈波あり
    false = [0] #脈拍なし
    noize = [2] #ノイズ
    # for i in range(num_segments):
    #     s = i* (segment_length - overlap)
    #     if  s + 15 <= time_rpulse[lavel_cunt - 1] <= s + 25 and labels[lavel_cunt - 1] == 1:
    #         start = i * (segment_length - overlap)
    #         end = start + segment_length
    #         x_data[i, :] = repulse[start:end]
    #         y_data[i, :] = true
    #         lavel_cunt += 1
    #         print('脈拍あり')
    #         print(lavel_cunt)
    #     elif s <= time_rpulse[lavel_cunt - 1] and time_rpulse[lavel_cunt] >= s + 50 and labels[lavel_cunt] == 0 and labels[lavel_cunt - 1] == 0:
    #         start = i * (segment_length - overlap)
    #         end = start + segment_length
    #         x_data[i, :] = repulse[start:end]
    #         y_data[i, :] = noize
    #         lavel_cunt += 1
    #         print('ノイズ')
    #         print(lavel_cunt)
    #     else:
    #         start = i * (segment_length - overlap)
    #         end = start + segment_length
    #         x_data[i, :] = repulse[start:end]
    #         y_data[i, :] = false
    #         # print('脈拍なし')

# def validation_noize(x_data, y_data):
    
    
def pridict_pulserate(x_data, y_data):
    # データをトレーニングセットとテストセットに分割
    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=0)
    # マルチラベル分類モデルを作成
    # classifier = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=0))
    classifier = RandomForestClassifier(n_estimators=100, random_state=0)
    # モデルを訓練
    classifier.fit(X_train, y_train)

    # テストデータで予測
    y_pred = classifier.predict(X_test)
    
    # 精度評価
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    # テストデータで混同行列を計算
    confusion = confusion_matrix(y_test, y_pred)

    # 混同行列を表示
    print("Confusion Matrix:")
    print(confusion)

    # 各クラスごとの評価指標を計算
    # 例: 適合率 (Precision), 再現率 (Recall), F1スコア (F1-Score)
    report = classification_report(y_test, y_pred)
    print("Classification Report:")
    print(report)
    
    filename = 'freq_study.sav'
    pickle.dump(classifier, open(filename, 'wb'))
    
    fig1 = plt.figure()
    fig1.set_figwidth(20)
    plt.plot(y_test)
    plt.plot(y_pred, "-.")
    plt.show()

    # # Jaccard Index (Intersection over Union) を計算
    # jaccard = jaccard_score(y_test, y_pred, average='samples')
    # print("Jaccard Index (Intersection over Union):", jaccard)

if __name__ == "__main__":    
    #初期値定義
    before_start = time.time()
    n = int(input('ファイル番号を指定してください:'))
    fs = 500
    t = np.arange(0, 600, 1/ fs)
    ecg_heart, ecg_pulse = fft_wave.read_data(n, fs, 1) #silver
    # ecg_heart, ecg_pulse = fft_wave.read_data(n, fs, 2) #file
    repulse = create_dataset(ecg_pulse, fs)
    th_pulse = auto_thread.auto_thread(repulse)
    time_rpulse = peek_pridict.get_peaks(repulse, th_pulse, 0, 300000)
    rpulse_diff = np.diff(time_rpulse)
    ave_peekdiff = np.mean(rpulse_diff)
    
    #################### ここまでは完成 #############################

    # ピークの削除
    indices_to_remove = []  # 削除するピークのインデックスを格納するリスト
    time_to_remove = []  # 削除するピークの時刻を格納するリスト
    for i in range(1, len(rpulse_diff) - 1):
        if np.abs(rpulse_diff[i] - ave_peekdiff) >= 100 and np.abs(rpulse_diff[i+1] - ave_peekdiff) >= 100:
            # 二回連続で差がしきい値以上の場合、ピークを削除する
            indices_to_remove.append(i+1)
            time_to_remove.append(time_rpulse[i+1])
            
            
    fig = plt.figure(figsize=(20, 10))    
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    ax1.plot(repulse, label="Original Waveform")
    ax1.plot(time_rpulse[0:len(repulse[0:len(repulse)])], repulse[time_rpulse[0:len(repulse[0:len(repulse)])]], "ro")
    
    k = 0
    for i in range(len(time_rpulse)):
        if time_rpulse[i] == time_to_remove[k]:
            k += 1
            for j in range(time_rpulse[i]-50, time_rpulse[i]+50):
                repulse[j] = repulse[j]/100
        if k == 5:
            break
                
    ax2.plot(repulse, label = 'Change Waveform')
    ax2.plot(time_rpulse[0:len(repulse[0:len(repulse)])], repulse[time_rpulse[0:len(repulse[0:len(repulse)])]], "ro")

    plt.show()
                        
    # 削除するピークを実際に削除
    # time_rpulse = np.delete(time_rpulse, indices_to_remove)
    # rpulse_diff = np.diff(time_rpulse)
    # print(rpulse_diff)
    # fig1 = plt.figure()
    # fig1.set_figwidth(20)
    # plt.plot(repulse)
    # plt.plot(time_rpulse[0:len(repulse[0:len(repulse)])], repulse[time_rpulse[0:len(repulse[0:len(repulse)])]], "ro")
    # plt.show()

    # ラベルリストを初期化し、すべてのピークを正常な脈拍（1）としてマーク
    # labels = np.ones(len(time_rpulse), dtype=int)

    # ピーク間隔時間と平均間隔の差を確認してノイズのピーク（0）を特定
    # for i in range(len(rpulse_diff)):
    #     if abs(rpulse_diff[i] - ave_peekdiff) > 300:
    #         labels[i] = 0
    #         labels[i + 1] = 0  # ノイズのピークを0（ノイズ）としてラベル付け
    
    #ノイズの中央時刻(time_to_remove)に対応するラベル(0)を再設定
    # k = 0
    # for i in range(len(time_rpulse)):
    #     if time_rpulse[i] == time_to_remove[k]:
    #         k += 1
    #         labels[i] = 0
    #     elif k == 5:
    #         print('変換完了')
    #         break
    
    # print(labels)
            

    # x_data = np.array(x_data)
    # y_data = np.array(y_data)
    # print("二次元データ:")
    # print(x_data)
    # print("教師用ラベルデータ:")
    # print(y_data)
    
    # pridict_pulserate(x_data, y_data)
    # tensor_study(x_data, y_data)
            
            
            
    