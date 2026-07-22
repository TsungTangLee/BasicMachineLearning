import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import pickle
import os

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

num_training = int(0.8*len(x))
num_test = len(x)-num_training
x_train = np.array(x[:num_training]).reshape((num_training, 1))
y_train = np.array(y[:num_training])
x_test = np.array(x[num_training:]).reshape((num_test, 1))
y_test = np.array(y[num_training:])
ridge_regressor = linear_model.Ridge(alpha=1)
ridge_regressor.fit(x_train, y_train)

y_test_pred = ridge_regressor.predict(x_test)
plt.figure()
plt.scatter(x_train, y_train, color='green')
plt.plot(x_test, y_test_pred, color='black', linewidth=4)
plt.title('Training data')
plt.show()

output_file = os.path.join(origin_path, "saved_ridge_model.pkl")
with open(output_file, 'wb') as f:
    pickle.dump(ridge_regressor, f)