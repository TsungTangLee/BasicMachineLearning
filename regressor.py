import os
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import pickle

# argv 擷取輸入的參數
origin_path = r"C:\Users\USER\Desktop\職前訓練"
filename = os.path.join(origin_path, "data.txt")

# 建立空列表
x = []
y = []

# 讀取檔案，並從每一行分別寫入x, y
with open(filename, 'r') as f:
    for line in f.readlines():
        print(line.split(','))
        xt, yt = [float(i) for i in line.split(',')]
        x.append(xt)
        y.append(yt)

# 分成訓練資料以及測試資料
num_training = int(0.8*len(x))
num_test = len(x)-num_training
x_train = np.array(x[:num_training]).reshape((num_training, 1))
y_train = np.array(y[:num_training])
x_test = np.array(x[num_training:]).reshape((num_test, 1))
y_test = np.array(y[num_training:])

# 線性回歸器做擬合
linear_regressor = linear_model.LinearRegression()
linear_regressor.fit(x_train, y_train)

# 使用擬合完得到的回歸直線做預測
y_test_pred = linear_regressor.predict(x_test)

# 畫圖
plt.figure()
plt.scatter(x_train, y_train, color='green')
plt.plot(x_test, y_test_pred, color='black', linewidth=4)
plt.title('Training data')
plt.show()

output_file = os.path.join(origin_path, "saved_model.pkl")
# 以寫入 byte 模式打開檔案
print("開始儲存模型...")
print(output_file)
with open(output_file, "wb") as f:
    # 將回歸器寫入
    pickle.dump(linear_regressor, f)

print("模型已儲存！")