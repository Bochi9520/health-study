from scipy import signal
import numpy as np
from scipy import fftpack
import csv
import matplotlib.pyplot as plt

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

silver = [
    "silver/female_76_safe.csv", #0
    "silver/female_76_free.csv", #1
    "silver/female_80_safe.csv", #2
    "silver/female_80_free.csv", #3
    "silver/female_81_safe.csv", #4
    "silver/female_81_free.csv", #5
    "silver/male_76_safe.csv", #6
    "silver/male_76_free.csv", #7
    "silver/male_77_safe.csv", #8
    "silver/male_77_free.csv", #09
    "silver/male_78_safe.csv", #10
    "silver/male_78_free.csv", #11
    "silver/male_79_safe.csv", #12
    "silver/male_79_free.csv", #13
]

def read_data(n, fs, kind): #kind 1=silver, 2=file
    if kind == 1:
        csv_file = open ("wave_file/"+silver[n], 'r', encoding="ms932", errors="", newline="")
    elif kind == 2:
        csv_file = open ("wave_file/"+file_name[n], 'r', encoding="ms932", errors="", newline="")
    f = csv.reader(csv_file, delimiter=",", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)
    ecg_heart = []
    ecg_pulse = []
    elbor_pulse = []
    leg_pulse = []
    period = 600
    t = np.arange(0, period, 1/fs)
    for row in f:
        ecg_heart.append(float(row[1]))
        ecg_pulse.append(float(row[2]))
        # ecg_pulse.append(float(row[7])) #wave_study_2用
        # leg_pulse.append(float(row[2])) #wave_study_2用
        
        #標準化したとき
        # ecg_heart.append(float(row[3]))
        # ecg_pulse.append(float(row[4]))
        
        # elbor_pulse.append(float(row[3]))
    
    csv_file.close()
        
    # return ecg_heart, ecg_pulse, elbor_pulse #PWV計測の時に使用
    return ecg_heart, ecg_pulse

def wave_diff(wave):
    wave = np.array(wave)
    wave = np.gradient(wave)
    return wave

def highpass(x, N, cutoff,samplerate): #データ,フィルタ次数,カットオフ周波数
    b, a = signal.butter(N, cutoff, "high",fs=samplerate)           #フィルタ伝達関数の分子と分母を計算
    y = signal.filtfilt(b, a, x)                  #信号に対してフィルタをかける
    return y   

def lowpass(x, N, cutoff,samplerate): #データ,フィルタ次数,カットオフ周波数
    b, a = signal.butter(N, cutoff, "low",fs=samplerate)           #フィルタ伝達関数の分子と分母を計算
    y = signal.filtfilt(b, a, x)                  #信号に対してフィルタをかける
    return y   

# from PIL import Image
#オーバーラップ処理
def ov(data, samplerate, Fs, overlap):
    Ts = len(data) / samplerate         #全データ長
    Fc = Fs / samplerate                #フレーム周期
    x_ol = Fs * (1 - (overlap/100))     #オーバーラップ時のフレームずらし幅
    N_ave = int((Ts - (Fc * (overlap/100))) / (Fc * (1-(overlap/100)))) #抽出するフレーム数（平均化に使うデータ個数）
 
    array = []      #抽出したデータを入れる空配列の定義
 
    #forループでデータを抽出
    for i in range(N_ave):
        ps = int(x_ol * i)              #切り出し位置をループ毎に更新
        array.append(data[ps:ps+Fs:1])  #切り出し位置psからフレームサイズ分抽出して配列に追加
    return array, N_ave                 #オーバーラップ抽出されたデータ配列とデータ個数を戻り値にする
 
#窓関数処理（ハニング窓）
def hanning(data_array, Fs, N_ave):
    han = signal.hann(Fs)        #ハニング窓作成
    acf = 1 / (sum(han) / Fs)   #振幅補正係数(Amplitude Correction Factor)
 
    #オーバーラップされた複数時間波形全てに窓関数をかける
    for i in range(N_ave):
        data_array[i] = data_array[i] * han #窓関数をかける
 
    return data_array, acf

#FFT処理
def fft_ave(data_array,samplerate, Fs, N_ave, acf):
    fft_array = []
    for i in range(N_ave):
        fft_array.append(acf*np.abs(fftpack.fft(data_array[i])/(Fs/2))) #FFTをして配列に追加、窓関数補正値をかけ、(Fs/2)の正規化を実施。
 
    fft_axis = np.linspace(0, samplerate, Fs)   #周波数軸を作成
    fft_array = np.array(fft_array)             #型をndarrayに変換
    fft_mean = np.sqrt(np.mean(fft_array ** 2, axis=0))       #全てのFFT波形のパワー平均を計算してから振幅値とする
    return fft_mean, fft_axis

def fft(data, samplerate, Fs, overlap):
  #Fs = 4096       フレームサイズ
  #overlap = 90    オーバーラップ率

  #作成した関数を実行：オーバーラップ抽出された時間波形配列
  time_array, N_ave = ov(data, samplerate, Fs, overlap)
  #作成した関数を実行：ハニング窓関数をかける
  time_array, acf = hanning(time_array, Fs, N_ave)
  #作成した関数を実行：FFTをかける
  fft_mean, fft_axis = fft_ave(time_array, samplerate, Fs, N_ave, acf)
  
  return fft_mean, fft_axis

def heart_image(x, range1, range2, n):
    fs = 500
    period = 600
    t = np.arange(0, period, 1/fs)
    fig1 = plt.figure()
    fig1.set_figwidth(20)
    plt.plot(t[range1:range2], ecg_heart[range1:range2], color = "red")
    plt.title(file_name[n]+'pulseimg')
    # plt.plot(t[range1:range2], x[range1:range2], "b")
    plt.show()
    
    fig1.savefig('img/heart'+file_name[n]+'.png')
    
def pulse_image(x, range1, range2, n):
    fs = 500
    period = 600
    t = np.arange(0, period, 1/fs)
    fig1 = plt.figure()
    fig1.set_figwidth(20)
    # plt.plot(t[range1:range2], ecg_heart[range1:range2], color = "red")
    plt.plot(t[range1:range2], x[range1:range2], "b")
    plt.title(file_name[n]+'pulseimg')
    plt.show()
    
    fig1.savefig('img/pulse'+file_name[n]+'.png')
    
def fft_image(x, range1, range2, n):
    fft_array = fft(x, 500, 4096, 90)
    fft_fig = plt.figure()
    fft_fig.set_figwidth(20)
    plt.plot(fft_array[1][range1:range2], fft_array[0][range1:range2], "b")
    plt.title(file_name[n]+'fftimg')
    plt.show()
    
    fft_fig.savefig('img/fft'+file_name[n]+'.png')

if __name__ == "__main__":
    n = int(input('ファイル番号を指定してください:'))
    ecg_heart, ecg_pulse = read_data(n)
    heart_image(ecg_heart, 40000, 45000, n)
    fft_image(ecg_heart, 10, 1000, n)
    pulse_image(ecg_pulse, 40000, 45000, n)
    fft_image(ecg_pulse, 0, 500, n)
    
    pass_pulse = highpass(ecg_pulse, 5, 0.5, 500)
    pass_pulse = lowpass(pass_pulse, 5, 45, 500)
    pulse_image(pass_pulse, 40000, 45000, n)
    fft_image(pass_pulse, 0, 500, n)
    

# turning_point = []
# for i in range(0, 300001, 5000):
#     turning_point.append(i)
# k = 1
# for i in range(1, len(turning_point), 4):
#     wave_image(turning_point[i-1], turning_point[i], 1)
#     wave_image(turning_point[i], turning_point[i+1], 2)
#     wave_image(turning_point[i+1], turning_point[i+2], 3)
#     wave_image(turning_point[i+2], turning_point[i+3], 4)
#     im1 = Image.open('img/im1.png')
#     im2 = Image.open('img/im2.png')
#     im3 = Image.open('img/im3.png')
#     im4 = Image.open('img/im4.png')
#     dst = Image.new('RGB', (im1.width, im1.height+im1.height+im1.height+im1.height))
#     dst.paste(im1, (0, 0))
#     dst.paste(im2, (0, im1.height))
#     dst.paste(im3, (0, im1.height+im1.height))
#     dst.paste(im4, (0, im1.height+im1.height+im1.height))
#     dst.save('img/compare_img'+ str(k)+'.png')
#     k += 1
