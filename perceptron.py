import numpy as np
import os

if not hasattr(np, 'Inf'):
    np.Inf = np.inf
if not hasattr(np, 'asfarray'):
    np.asfarray = lambda a, dtype=float: np.asarray(a, dtype=dtype)

import neurolab as nl
import matplotlib.pyplot as plt

# 產生數據及標籤
data = np.array([[0.3, 0.2], [0.1, 0.4], [0.4, 0.6], [0.9, 0.5]])
labels = np.array([[0], [0], [0], [1]])

# 畫圖看數據
plt.figure()
plt.scatter(data[:,0], data[:,1])
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Input data')
plt.show()

# 呼叫感知器，第一個變數表示輸入層的大小(2)以及上下限，第二個變數表示輸出層的數量，預設的激勵函數為HardLim
perceptron = nl.net.newp([[0, 1],[0, 1]], 1)

# 將資料輸入以訓練感知器，同時會回傳誤差值(Loss)，誤差預設為sse，epochs表示訓練次數上限，lr表示學習速率
error = perceptron.train(data, labels, epochs=50, show=10, lr=0.01)

# 畫圖顯示誤差變化
plt.figure()
plt.plot(error)
plt.xlabel('Number of epochs')
plt.ylabel('Training error')
plt.grid()
plt.title('Training error progress')
plt.show()

# 測試感知器
test_data = np.array([[0.9, 0.5]])
print(perceptron.sim(test_data))