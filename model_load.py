import os
import pickle
import numpy as np

# 指定讀取檔案位置
origin_path = r"C:\Users\USER\Desktop\職前訓練"
read_file = os.path.join(origin_path, "saved_model.pkl")
# 以讀取byte模式打開檔案
with open(read_file, 'rb') as f:
    model_linregr = pickle.load(f)
# 測試資料
X_test = np.array([1,2,3,4]).reshape((-1,1))
y_test_pred_new = model_linregr.predict(X_test)
print(y_test_pred_new)