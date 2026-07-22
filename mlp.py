import numpy as np
if not hasattr(np, 'Inf'):
    np.Inf = np.inf
if not hasattr(np, 'asfarray'):
    np.asfarray = lambda a, dtype=float: np.asarray(a, dtype=dtype)
import neurolab as nl
import matplotlib.pyplot as plt
import os

# 定義數據最大最小值、大小
min_value = -12
max_value = 12
num_datapoints = 90

# 產生x,y數據
x = np.linspace(min_value, max_value, num_datapoints)
y = 2 * np.square(x) + 7
y /= np.linalg.norm(y)

# 將數據轉化成可以輸入神經網路的格式
data = x.reshape(num_datapoints, 1)
labels = y.reshape(num_datapoints, 1)

plt.figure()
plt.scatter(data, labels)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Input data')
plt.show()

# 定義激勵函數
transfun =  [nl.trans.TanSig(), nl.trans.TanSig(), nl.trans.TanSig()]

# 呼叫多層神經網路，第二個變數定義除了輸入層外的神經網路形狀
multilayer_net = nl.net.newff([[min_value, max_value]], [20, 20, 1], 	transf = transfun)

# 利用梯度下降法訓練
multilayer_net.trainf = nl.train.train_gd

# 開始訓練並回傳誤差
error = multilayer_net.train(data, labels, epochs=800, show=100, goal=0.01)

# 產生預測值
predicted_output = multilayer_net.sim(data)

# 畫圖顯示訓練過程
plt.figure()
plt.plot(error)
plt.xlabel('Number of epochs')
plt.ylabel('Error')
plt.title('Training error progress')
plt.show()

# 畫圖比較實際值、預測值
x2 = np.linspace(min_value, max_value, num_datapoints * 2)
y2 = multilayer_net.sim(x2.reshape(x2.size,1)).reshape(x2.size)
y3 = predicted_output.reshape(num_datapoints)
plt.figure()
plt.plot(x2, y2, '-', x, y, '.', x, y3, 'p')
plt.title('Ground truth vs predicted output')
plt.show()