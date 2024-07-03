
# 数理最適化の練習
# 参考
# [1]並木誠, Pythonによる数理最適化, 朝倉書店, 2018
# [2]梅谷俊治, しっかり学ぶ数理最適化, 講談社, 2021


#%%
# [1] chapter5
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-3, 3)
f1 = lambda x: 2*x

plt.plot(x, f1(x), color='k', linestyle='-')
plt.show()


# %%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x = y = np.linspace(-2, 2)
x, y = np.meshgrid(x, y)

g1 = lambda x: 2*x[0] - 3*x[1] # x = (x[0], x[1], x[2], ...)という感じのベクトル表示になっている

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(x, y, g1([x, y]), rstride=2, cstride=2)
plt.show()


# %%
plt.contourf(x, y, g1([x, y]), cmap=plt.cm.binary)


# %%
from sympy import * # sympyモジュールからすべての関数やオブジェクトをインポートしている
x = [Symbol('x[0]'), Symbol('x[1]')] # x[0], x[1]を文字扱いする

f1 = lambda x: (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2

print([diff(f1(x), x[0]), diff(f1(x), x[1])]) # 勾配ベクトル
print([[diff(f1(x), x[0], x[0]), diff(f1(x), x[0], x[1])],
       [diff(f1(x), x[1], x[0]), diff(f1(x), x[1], x[1])]]) # ヘッセ行列


# %%
# scipyを用いたコレスキー分解の例
import numpy as np
import scipy.linalg as linalg
a = np.random.randint(-10, 10, (3,2))
A = np.dot(a.T, a)
print(A)
U = linalg.cholesky(A)
print(U)
print(np.dot(U.T, U))


# %%
# ニュートン法によるRosenbrock関数の非線形最適化
import numpy as np
import scipy.linalg as linalg
from sympy import *

f = lambda x: sum(100*(x[i+1] - x[i]**2)**2 + (x[i] - 1)**2 for i in range(2))

nf = lambda x: np.array([-400*x[0]*(-x[0]**2 + x[1]) + 2*x[0] - 2, 
                         -200*x[0]**2 - 400*x[1]*(-x[1]**2 + x[2]) + 202*x[1] - 2,
                         -200*x[1]**2 + 200*x[2]])

Hf = lambda x: np.array([[1200*x[0]**2 - 400*x[1] + 2, -400*x[0], 0],
                         [-400*x[0], 1200*x[1]**2 - 400*x[2] + 202, -400*x[1]],
                         [0, -400*x[1], 200]])

x0 = [10, 10, 10]
MEPS = 1.0e-6

k = 0
while linalg.norm(nf(x0)) > MEPS:
    d = -np.dot(linalg.inv(Hf(x0)), nf(x0))
    x0 = x0 + d
    k = k + 1

print('interation:', k)
print('optimal solution:', x0)


# %%
# sympyを用いてな勾配ベクトルとヘッセ行列の計算を自動化する
import numpy as np
import scipy.linalg as linalg
from sympy import *

# 変数の定義
x = symbols('x0:3')

# Rosenbrock関数の定義
f = sum(100*(x[i+1] - x[i]**2)**2 + (x[i] - 1)**2 for i in range(2))

# Rosenbrock関数の勾配の定義
nf = [diff(f, xi) for xi in x]

# Rosenbrock関数のヘッセ行列の定義
Hf = [[diff(nfj, xi) for xi in x] for nfj in nf]

# 数値化した関数と行列
f_func = lambdify(x, f, 'numpy')
nf_func = lambdify(x, nf, 'numpy')
Hf_func = lambdify(x, Hf, 'numpy')

# 初期点
x0 = np.array([10, 10, 10])
MEPS = 1.0e-6

k = 0
while linalg.norm(nf_func(*x0)) > MEPS:
    d = -np.dot(linalg.inv(Hf_func(*x0)), nf_func(*x0))
    x0 = x0 + d
    k += 1

print('iteration:', k)
print('optimal solution:', x0)

# %%
