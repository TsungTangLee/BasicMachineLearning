from sklearn import preprocessing
import numpy as np
# 資料
data = np.array([[1,2,3,4],
                 [5,6,7,8],
                 [9,10,11,12]])
# 平均值以及標準差
b = data.mean(axis=0)
a = data.std(axis=0)
print(b)
print(a)
c = (data-b)/a
print(c)

# 標準化，將(資料-b)/a，b為平均值，a為標準差。
data_standardized = preprocessing.scale(data, axis=0)
print("數據標準化:\n", data_standardized)
print("平均值:\n", data_standardized.mean(axis=0))
print("標準差:\n", data_standardized.std(axis=0))

# 範圍縮放，將資料限制在0~1
data_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
data_scaled = data_scaler.fit_transform(data)
print("數據縮放:\n", data_scaled)

# 歸一化
data_normalized = preprocessing.normalize(data, norm='l1')
print("數據歸一化:\n", data_normalized)

# 二元化
data_binarized = preprocessing.Binarizer(threshold=1).transform(data)
print("數據二元化:\n", data_binarized)

# 標籤編碼
label_encoder = preprocessing.LabelEncoder()
input_classes = ['apple', 'banana', 'grape', 'orange', 'apple']
label_encoder.fit(input_classes)
print("標籤分類:", label_encoder.classes_)
labels = ['banana', 'apple']
encoded_labels = label_encoder.transform(labels)
print("標籤編碼:", encoded_labels)
decoded_labels = label_encoder.inverse_transform(encoded_labels)
print("標籤解碼:", decoded_labels)