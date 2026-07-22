from sklearn.preprocessing import PolynomialFeatures
import sklearn.metrics as sm
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import pickle

filename = "data.txt" 
def show_metrics(y, y_pred):
    print("平均絕對值誤差(Mean absolute error):", round(sm.mean_absolute_error(y, y_pred), 2))
    print("平均平方誤差(Mean squared error):", round(sm.mean_squared_error(y, y_pred), 2))
    print("中值絕對離差(Median absolute error):", round(sm.median_absolute_error(y, y_pred), 2))
    print("解釋方差分(Explained variance score):", round(sm.explained_variance_score(y, y_pred), 2))
    print("決定係數(R2 score):", round(sm.r2_score(y, y_pred), 2))

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

# 擷取多項式特徵
polynomial = PolynomialFeatures(degree=3)  
# 轉換訓練資料為多項式
x_train_transformed = polynomial.fit_transform(x_train)  
print(x_train_transformed)
# 線性回歸器做擬合
linear_regressor = linear_model.LinearRegression()
linear_regressor.fit(x_train_transformed, y_train) 

# 指定輸出檔案位置
output_file = 'saved_polynomial_model.pkl'
# 以寫入byte模式打開檔案
with open(output_file, 'wb') as f:
    # 將回歸器寫入
    pickle.dump(linear_regressor, f)

# 轉換輸入資料為多項式
x_test_transformed = polynomial.fit_transform(x_test)   
# 使用擬合完得到的回歸直線做預測
y_train_pred = linear_regressor.predict(x_train_transformed)

# 畫圖
plt.figure()
plt.scatter(x_train, y_train, color='green')
plt.scatter(x_train, y_train_pred, color='black')
plt.title('Training data')
plt.show()

# 換成對測試資料做預測
y_test_pred = linear_regressor.predict(x_test_transformed) 

plt.figure()
plt.scatter(x_test, y_test, color='green')
plt.scatter(x_test, y_test_pred, color='black')
plt.title('Testing data')
plt.show()

show_metrics(y_test, y_test_pred)

# 取得多項式轉換後，每個特徵欄位的名稱
feature_names = polynomial.get_feature_names_out(['x'])

# 取得模型學到的權重（斜率）與截距
weights = linear_regressor.coef_
intercept = linear_regressor.intercept_

print("===== 模型權重回傳 =====")
print(f"截距 (Intercept / Bias): {intercept:.4f}")
for name, weight in zip(feature_names, weights):
    print(f"特徵 {name} 的權重 (Weight): {weight:.4f}")