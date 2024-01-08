import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

import fft_wave


def create_windowdata(ecg, pulse, num): #脈波系データ, 心電図データ, 窓幅
    train_pulse = np.array(pulse)
    train_heart = np.array(ecg)
    pulse_X = np.empty((len(pulse),num))
    ecg_X = np.empty((len(pulse),num))
    
    for i in range(num):
        pulse_X[:, i] = np.roll(train_pulse, -i)
        ecg_X[:, i] = np.roll(train_heart, -i)
        
    pulse_X = pulse_X[num-1: -num+1]
    ecg_X = ecg_X[num-1: -num+1]
    
    print(pulse_X.shape)
    print(ecg_X.shape)
        
    return ecg_X, pulse_X

def pulse_NN(ecg_data, pulse_data):
    # 脈波データ（300000サンプル、10分、サンプリング周波数 500）
    # 心電図データ（300000サンプル、10分、サンプリング周波数 500）
    num_samples = len(ecg_data)
    num_data_points_ecg = ecg_data.shape[1]
    num_data_points_pulse = pulse_data.shape[1]
    # 訓練データとテストデータに分割
    train_ratio = 0.8
    num_train_samples = int(num_samples * train_ratio)

    train_pulse = pulse_data[:num_train_samples]
    train_ecg = ecg_data[:num_train_samples]

    test_pulse = pulse_data[num_train_samples:]
    test_ecg = ecg_data[num_train_samples:]

    # ニューラルネットワークの構築
    model = keras.Sequential([
        keras.layers.Input(shape=(num_data_points_pulse,)),  # 脈波の次元数を指定
        keras.layers.Dense(128, activation='relu'),  # 中間層
        keras.layers.Dense(num_data_points_ecg, activation='linear')  # 出力層（心電図の次元数を指定）
    ])

    # モデルのコンパイル
    model.compile(optimizer='adam', loss='mean_squared_error')

    # モデルの訓練
    num_epochs = 10
    model.fit(train_pulse, train_ecg, epochs=num_epochs, validation_data=(test_pulse, test_ecg))

    # テストデータでの予測
    predicted_ecg = model.predict(test_pulse)

    return test_ecg, predicted_ecg

# 予測された心電図と実際の心電図をプロットして比較
if __name__ == "__main__":
    n = int(input('使用するデータを指定してください: '))
    ecg, pulse = fft_wave.read_data(n, 500, 2)
    ecg = fft_wave.lowpass(ecg, 5, 20, 500)
    pulse = fft_wave.lowpass(pulse, 5, 20, 500)
    ecg_data, pulse_data = create_windowdata(ecg, pulse, 300)
    test_ecg, predicted_ecg = pulse_NN(ecg_data, pulse_data)
    plt.plot(test_ecg[0], label='Actual ECG')
    plt.plot(predicted_ecg[0], label='Predicted ECG')
    plt.legend()
    plt.show()
