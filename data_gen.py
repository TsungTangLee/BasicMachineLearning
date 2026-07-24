import random
# x: 0~10 等間隔排列
x_list = [i*0.1 for i in range(100)]
# y: 3.3x + -1~1
y_list = [x_list[i]*3.3+x_list[i]**3+random.uniform(-1, 1) for i in range(100)]

# 存檔路徑
path = "data.txt"
f = open(path, 'w')
# 跑回圈寫入每一行
for i in range(100):
    print(x_list[i],",", y_list[i], file=f)
f.close()
