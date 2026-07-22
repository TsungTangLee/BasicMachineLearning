import numpy as np
from numpy import pi, cos, sin, sqrt, exp
import matplotlib.pyplot as plt
import matplotlib as mpl
import imageio.v2 as imageio
import os

output_dir = './fig'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# --- 1. 畫圖與環境設定 ---
mpl.rcParams['figure.dpi'] = 100
mpl.rcParams['figure.figsize'] = [10, 5]  # [x,y]
mpl.rcParams['font.size'] = 13

f_ = lambda x: (1*x**4 - 50*x**3 - 1*x + 1)
x = np.linspace(-30, 60, 1000)
x_move = -18
gamma = 0.01
dx = 0.01

plt.ioff()  # 關閉交互模式，避免螢幕彈出 100 個視窗
filenames = []

for index, i in enumerate(x[:100]):
    pattern = f_(x)
    plt.plot(x, pattern, c='gray', label='target function')
    plt.scatter(x_move, f_(x_move), c='red', label=r'$ x_{s}$')
    
    df_ = (f_(x_move+dx) - f_(x_move))/dx
    x_move = x_move - 100*gamma if df_ > 0 else x_move + 100*gamma
    text = 'iterative times:' + str(index) + '\n' + r'$-\gamma df_(x) =$' + str(np.round(-gamma*df_, 2))
    text += ' x= ' + str(np.round(x_move, 2))
    plt.title(text)
    plt.grid(True, linestyle='-.')
    plt.ylim(-1*10**(6), 2*10**(6))
    plt.xlim(-30, 60)
    plt.ylabel('Amplitude')
    plt.xlabel('x')
    
    if index == 0:
        plt.legend()
    current_filename = './fig/test' + 'T=' + "{:.2f}".format(np.round(i, 2)) + '.jpg'
    plt.savefig(current_filename)
    plt.close()
    filenames.append(current_filename)

images = []
for i in filenames:
    if os.path.exists(i):  # 確保檔案存在才讀取
        images.append(imageio.imread(i))
        os.remove(i)       # 讀取完順手刪除臨時圖檔，保持資料夾乾淨

imageio.mimsave('./code-movie-gradient descent.gif', images)