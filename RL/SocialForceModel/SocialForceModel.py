"""
 ▄▄▄▄▄▄▄▄▄▄▄  ▄            ▄▄▄▄▄▄▄▄▄▄▄  ▄▄        ▄ 
▐░░░░░░░░░░░▌▐░▌          ▐░░░░░░░░░░░▌▐░░▌      ▐░▌
▐░█▀▀▀▀▀▀▀█░▌▐░▌          ▐░█▀▀▀▀▀▀▀█░▌▐░▌░▌     ▐░▌
▐░▌       ▐░▌▐░▌          ▐░▌       ▐░▌▐░▌▐░▌    ▐░▌
▐░█▄▄▄▄▄▄▄█░▌▐░▌          ▐░█▄▄▄▄▄▄▄█░▌▐░▌ ▐░▌   ▐░▌
▐░░░░░░░░░░░▌▐░▌          ▐░░░░░░░░░░░▌▐░▌  ▐░▌  ▐░▌
▐░█▀▀▀▀▀▀▀▀▀ ▐░▌          ▐░█▀▀▀▀▀▀▀█░▌▐░▌   ▐░▌ ▐░▌
▐░▌          ▐░▌          ▐░▌       ▐░▌▐░▌    ▐░▌▐░▌
▐░▌          ▐░█▄▄▄▄▄▄▄▄▄ ▐░▌       ▐░▌▐░▌     ▐░▐░▌
▐░▌          ▐░░░░░░░░░░░▌▐░▌       ▐░▌▐░▌      ▐░░▌
 ▀            ▀▀▀▀▀▀▀▀▀▀▀  ▀         ▀  ▀        ▀▀ 
"""
# 参考文献
#[1]安藤歩, ソーシャルフォースモデルを用いた教室内における避難行動シミュレーション
#[2]安田＆水野, 火災避難を対象としたマルチエージェントシミュレーション
#[3]磯崎＆中辻, Social force modelを元にした歩行者の避難シミュレーションモデルに関する研究         
#[4]@ReoNagai, SocailForceModelの解説と実装, https://qiita.com/ReoNagai/items/5f94aecec4c3e7fa9135                      

"""
 ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄   ▄▄▄▄▄▄▄▄▄▄▄     ▄▄▄▄     
▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░▌ ▐░░░░░░░░░░░▌  ▄█░░░░▌    
▐░█▀▀▀▀▀▀▀▀▀ ▐░█▀▀▀▀▀▀▀█░▌▐░█▀▀▀▀▀▀▀█░▌▐░█▀▀▀▀▀▀▀▀▀  ▐░░▌▐░░▌    
▐░▌          ▐░▌       ▐░▌▐░▌       ▐░▌▐░▌            ▀▀ ▐░░▌    
▐░▌          ▐░▌       ▐░▌▐░▌       ▐░▌▐░█▄▄▄▄▄▄▄▄▄      ▐░░▌    
▐░▌          ▐░▌       ▐░▌▐░▌       ▐░▌▐░░░░░░░░░░░▌     ▐░░▌    
▐░▌          ▐░▌       ▐░▌▐░▌       ▐░▌▐░█▀▀▀▀▀▀▀▀▀      ▐░░▌    
▐░▌          ▐░▌       ▐░▌▐░▌       ▐░▌▐░▌               ▐░░▌    
▐░█▄▄▄▄▄▄▄▄▄ ▐░█▄▄▄▄▄▄▄█░▌▐░█▄▄▄▄▄▄▄█░▌▐░█▄▄▄▄▄▄▄▄▄  ▄▄▄▄█░░█▄▄▄ 
▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░▌ ▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌
 ▀▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀   ▀▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀▀
"""

#%%
# social force model
# 参考：[1], [2]

import numpy as np

class SFM:
    def __init__(self, num, mass, x_start, y_start, x_goal, y_goal, v, walls, A, B, r, k, kappa):
        self.num = num
        self.mass = mass
        self.x_start = x_start
        self.y_start = y_start
        self.x_goal = x_goal
        self.y_goal = y_goal
        self.v = v
        self.walls = walls
        self.A = A
        self.B = B
        self.r = r
        self.k = k
        self.kappa = kappa

    def vector_normalize(self, x, y):
        xy = np.column_stack((x, y))
        xy_norm = np.linalg.norm(xy, axis=1)
        # Avoid division by zero by checking if the norm is non-zero
        xy_normalize = np.divide(xy, xy_norm[:, None], out=np.zeros_like(xy, dtype=float), where=(xy_norm[:, None] != 0))
        return xy_normalize[:, 0], xy_normalize[:, 1]

    def f1(self, x, y):
        dx = self.x_goal - x
        dy = self.y_goal - y
        dx_normal, dy_normal = self.vector_normalize(dx, dy)
        f1_x = self.v * dx_normal
        f1_y = self.v * dy_normal
        return f1_x, f1_y
    
    def f2(self, x, y):
        dx = x[np.newaxis, :] - x[:, np.newaxis] # col is n_ij = j - i
        dy = y[np.newaxis, :] - y[:, np.newaxis]
        dist = np.sqrt(dx**2 + dy**2)

        f2_x = []
        f2_y = []
        for i in range(self.num):
            t_ij_x = -dy[i]
            t_ij_y = dx[i]
            dist_i = dist[i]
            r_ij = self.r + self.r[i]
            theta = np.where(r_ij - dist_i < 0, 0, 1)
            f_ij_x = (self.A[i] * np.exp((r_ij - dist_i) / self.B[i]) + self.k * theta) * dx[i] + self.kappa * theta * t_ij_x**2
            f_ij_y = (self.A[i] * np.exp((r_ij - dist_i) / self.B[i]) + self.k * theta) * dy[i] + self.kappa * theta * t_ij_y**2

            walls_x = self.walls[:,0]
            walls_y = self.walls[:,1]
            n_iw_x = walls_x - x[i]
            n_iw_y = walls_y - y[i]
            d_iw = np.sqrt(n_iw_x**2 + n_iw_y**2)
            theta_iw = np.where(self.r[i] - d_iw < 0, 0, 1)
            t_iw_x = -n_iw_x
            t_iw_y = n_iw_x

            f_iw_x = (self.A[i] * np.exp((self.r[i] - d_iw) / self.B[i]) + self.k * theta_iw) * n_iw_x - self.kappa * theta_iw * self.v * t_iw_x**2
            f_iw_y = (self.A[i] * np.exp((self.r[i] - d_iw) / self.B[i]) + self.k * theta_iw) * n_iw_y - self.kappa * theta_iw * self.v * t_iw_y**2

            f2_x_i = np.sum(f_ij_x) + np.sum(f_iw_x)
            f2_y_i = np.sum(f_ij_y) + np.sum(f_iw_y)

            f2_x.append(f2_x_i)
            f2_y.append(f2_y_i)
        
        return np.array(f2_x), np.array(f2_y)
    
    def step(self, x, y):
        f1_x, f1_y = self.f1(x, y)
        f2_x, f2_y = self.f2(x, y)

        f_x = f1_x + (1 / self.mass) * f2_x
        f_y = f1_y + (1 / self.mass) * f2_y

        x_next = x + f_x
        y_next = y + f_y
        return x_next, y_next
    
    def reset(self):
        x = self.x_start
        y = self.y_start
        return x, y


#%%
# 実行プログラム
import numpy as np

class Run:
    def __init__(self, Tmax, num, mass, x_start, y_start, x_goal, y_goal, v, walls, A, B, r, k, kappa):
        self.Tmax = Tmax
        self.num = num
        self.mass = mass
        self.x_start = x_start
        self.y_start = y_start
        self.x_goal = x_goal
        self.y_goal = y_goal
        self.v = v
        self.walls = walls
        self.A = A
        self.B = B
        self.r = r
        self.k = k
        self.kappa = kappa

    def run_sim(self):
        agents = SFM(self.num, self.mass, self.x_start, self.y_start, self.x_goal, self.y_goal, self.v, self.walls, self.A, self.B, self.r, self.k, self.kappa)
        x_record = []
        y_record = []
        x, y = agents.reset()
        x_record.append(x)
        y_record.append(y)
        for t in range(self.Tmax):
            x_next, y_next = agents.step(x, y)
            x, y = x_next, y_next
            x_record.append(x)
            y_record.append(y)
        return x_record, y_record


#%%
# 実行
import numpy as np
import random

Tmax = 50
num = 100
mass = 60
x_start = np.random.uniform(low=0, high=50, size=num)
y_start = np.random.uniform(low=0, high=10, size=num)
x_goal = 100
y_goal = 50
v = 0.6
walls = np.array([[0,0], [0,10], [100,0], [60,10]])
A = np.ones(num)
B = np.ones(num)
r = np.ones(num) * 0.4
k = 1
kappa = 2
x_record = []
y_record = []

simulator = Run(Tmax, num, mass, x_start, y_start, x_goal, y_goal, v, walls, A, B, r, k, kappa)
x_record, y_record = simulator.run_sim()


#%%
# 描画
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML

# キャンバスを設置
fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot()

ax1.set_xlim(-10, 110)
ax1.set_ylim(-10, 110)

#アップデート関数
def update1(i):
    ax1.clear()
        
    ax1.set_xlim(-10, 110)
    ax1.set_ylim(-10, 110)

    data_x = x_record[i]
    data_y = y_record[i]

    ax1.scatter(data_x, data_y, color='black')
    ax1.hlines(0, 0, 60, color='blue')
    ax1.hlines(10, 0, 60, color='blue')
    # 進捗状況を出力する
    progress_rate = (i+1)/Tmax*100
    print("Animation progress={:.3f}%".format(progress_rate), end='\r')
        
#アニメーション作成とgif保存
ani = animation.FuncAnimation(fig, update1, frames=range(Tmax))
    
#グラフ表示
plt.show()

#アニメーションを表示
HTML(ani.to_jshtml())


#%%
"""
 ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄   ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄ 
▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░▌ ▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌
▐░█▀▀▀▀▀▀▀▀▀ ▐░█▀▀▀▀▀▀▀█░▌▐░█▀▀▀▀▀▀▀█░▌▐░█▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀█░▌
▐░▌          ▐░▌       ▐░▌▐░▌       ▐░▌▐░▌                    ▐░▌
▐░▌          ▐░▌       ▐░▌▐░▌       ▐░▌▐░█▄▄▄▄▄▄▄▄▄           ▐░▌
▐░▌          ▐░▌       ▐░▌▐░▌       ▐░▌▐░░░░░░░░░░░▌ ▄▄▄▄▄▄▄▄▄█░▌
▐░▌          ▐░▌       ▐░▌▐░▌       ▐░▌▐░█▀▀▀▀▀▀▀▀▀ ▐░░░░░░░░░░░▌
▐░▌          ▐░▌       ▐░▌▐░▌       ▐░▌▐░▌          ▐░█▀▀▀▀▀▀▀▀▀ 
▐░█▄▄▄▄▄▄▄▄▄ ▐░█▄▄▄▄▄▄▄█░▌▐░█▄▄▄▄▄▄▄█░▌▐░█▄▄▄▄▄▄▄▄▄ ▐░█▄▄▄▄▄▄▄▄▄ 
▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░▌ ▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌
 ▀▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀   ▀▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀▀                                                               
"""

#%%
# social force model2
# 参考：[4]
import numpy as np

class SFM2:
    def __init__(self, num, v, tau, R, U, start, goal, obstacles):
        self.num = num
        self.v = v
        self.tau = tau
        self.R = R
        self.U = U
        self.x_start = start[:,0]
        self.y_start = start[:,1]
        self.x_goal = goal[:,0]
        self.y_goal = goal[:,1]
        self.x_obstacles = obstacles[:,0]
        self.y_obstacles = obstacles[:,1]
        self.mass_kg = 55
        self.mass_N = self.mass_kg / 9.8
        self.delta_sec = 0.01

    def vector_normalize(self, x, y):
        xy = np.column_stack((x, y))
        xy_norm = np.linalg.norm(xy, axis=1)
        # Avoid division by zero by checking if the norm is non-zero
        xy_normalize = np.divide(xy, xy_norm[:, None], out=np.zeros_like(xy, dtype=float), where=(xy_norm[:, None] != 0))
        return xy_normalize[:, 0], xy_normalize[:, 1]

    def f_goal(self, x, y, v_x, v_y):
        print('x_goal={}'.format(self.x_goal))
        dx_goal = self.x_goal - x
        print('dx_goal={}'.format(dx_goal))
        dy_goal = self.y_goal - y
        dx_goal_norm, dy_goal_norm = self.vector_normalize(dx_goal, dy_goal)

        f_x = (self.v * dx_goal_norm - v_x) / self.tau
        f_y = (self.v * dy_goal_norm - v_y) / self.tau
        return f_x, f_y
    
    def f_obstacle(self, x, y):
        f_x = []
        f_y = []
        for i in range(self.num):
            x_i, y_i = x[i], y[i]
            dx = x_i - self.x_obstacles
            dy = y_i - self.y_obstacles
            dist = np.sqrt(dx**2 + dy**2)
            f_x_i = np.sum(dx[i] * self.U * np.exp(-dist / self.R))
            f_y_i = np.sum(dy[i] * self.U * np.exp(-dist / self.R))
            f_x.append(f_x_i)
            f_y.append(f_y_i)
        return np.array(f_x), np.array(f_y)
    
    def f_other(self, x, y):
        dx = x[np.newaxis, :] - x[:, np.newaxis]
        dy = y[np.newaxis, :] - y[:, np.newaxis]
        dist = np.sqrt(dx**2 + dy**2)
        f_x = np.sum(dx * self.U * np.exp(-dist / self.R), axis = 1)
        f_y = np.sum(dy * self.U * np.exp(-dist / self.R), axis = 1)
        return f_x, f_y
    
    def reset(self):
        x = self.x_start
        y = self.y_start
        vx = np.zeros(self.num)
        vy = np.zeros(self.num)
        return x, y, vx, vy

    def step(self, x, y, vx, vy):
        fx_goal, fy_goal = self.f_goal(x, y, vx, vy)
        fx_obst, fy_obst = self.f_obstacle(x, y)
        fx_other, fy_other = self.f_other(x, y)

        fx = fx_goal + fx_obst + fx_other
        fy = fy_goal + fy_obst + fy_other

        acc_x = fx / self.mass_N
        acc_y = fy / self.mass_N

        x_next = x + acc_x * self.delta_sec
        y_next = y + acc_y * self.delta_sec
        return x_next, y_next, fx, fy


#%%
# 実行関数
import numpy as np

class Run2:
    def __init__(self, Tmax, num, v, tau, R, U, start, goal, obstacles):
        self.Tmax = Tmax
        self.num = num
        self.v = v
        self.tau = tau
        self.R = R
        self.U = U
        self.start = start
        self.goal = goal
        self.obstacles = obstacles

    def run_sim(self):
        agent = SFM2(self.num, self.v, self.tau, self.R, self.U, self.start, self.goal, self.obstacles)
        x_history = []
        y_history = []
        vx_history = []
        vy_history = []

        x, y, vx, vy = agent.reset()
        x_history.append(x)
        y_history.append(y)
        vx_history.append(vx)
        vy_history.append(vy)

        for t in range(self.Tmax):
            x_next, y_next, vx_next, vy_next = agent.step(x, y, vx, vy)
            x, y, vx, vy = x_next, y_next, vx_next, vy_next
            x_history.append(x)
            y_history.append(y)
            vx_history.append(vx)
            vy_history.append(vy)
            progress_rate = (t+1)/self.Tmax*100
            print("progress={:.3f}%".format(progress_rate), end='\r')
        return x_history, y_history, vx_history, vy_history


#%%
# 実行
import numpy as np

# 諸定数
Tmax = 100
num = 1
v = 1.34
tau = 0.5
R = 0.2
U = 10.0
start = np.array([[1,1]])
goal = np.array([[20,20]])
obstacles = np.array([[9,9]])

simulator = Run2(Tmax, num, v, tau, R, U, start, goal, obstacles)

x, y, vx, vy = simulator.run_sim()



# %%
# アニメーションの描画
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML

maze = np.array([[]])

# キャンバスを設置
fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot()

ax1.set_xlim(-10, 30)
ax1.set_ylim(-10, 30)

#アップデート関数
def update1(i):
    ax1.clear()
        
    ax1.set_xlim(-10, 30)
    ax1.set_ylim(-10, 30)

    data_x = x[i]
    data_y = y[i]

    ax1.scatter(data_x, data_y, color='black')
    ax1.scatter(obstacles[:,0], obstacles[:,1], color='red')
    ax1.scatter(goal[:,0], goal[:,1], color='green')
    # 進捗状況を出力する
    progress_rate = (i+1)/Tmax*100
    print("Animation progress={:.3f}%".format(progress_rate), end='\r')
        
#アニメーション作成とgif保存
ani = animation.FuncAnimation(fig, update1, frames=range(Tmax))
    
#グラフ表示
plt.show()

#アニメーションを表示
HTML(ani.to_jshtml())


# %%
# 別のプログラム
"""
 ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄   ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄ 
▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░▌ ▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌
▐░█▀▀▀▀▀▀▀▀▀ ▐░█▀▀▀▀▀▀▀█░▌▐░█▀▀▀▀▀▀▀█░▌▐░█▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀█░▌
▐░▌          ▐░▌       ▐░▌▐░▌       ▐░▌▐░▌                    ▐░▌
▐░▌          ▐░▌       ▐░▌▐░▌       ▐░▌▐░█▄▄▄▄▄▄▄▄▄           ▐░▌
▐░▌          ▐░▌       ▐░▌▐░▌       ▐░▌▐░░░░░░░░░░░▌ ▄▄▄▄▄▄▄▄▄█░▌
▐░▌          ▐░▌       ▐░▌▐░▌       ▐░▌▐░█▀▀▀▀▀▀▀▀▀ ▐░░░░░░░░░░░▌
▐░▌          ▐░▌       ▐░▌▐░▌       ▐░▌▐░▌          ▐░█▀▀▀▀▀▀▀▀▀ 
▐░█▄▄▄▄▄▄▄▄▄ ▐░█▄▄▄▄▄▄▄█░▌▐░█▄▄▄▄▄▄▄█░▌▐░█▄▄▄▄▄▄▄▄▄ ▐░█▄▄▄▄▄▄▄▄▄ 
▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░▌ ▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌
 ▀▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀   ▀▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀▀                                                               
"""
# 引用：[4]
# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt

V0  = 1.34 # [m/s] Speed of equilibrium (= max speed)
tau = 0.5  # [sec] Time to reach V0 ziteisuu
delta_sec = 1 #[sec]

U = 10.0 # [m^2/s^2]
R = 0.2  # [m]

user_mass_kg = 55 # [kg]
user_mass_N = user_mass_kg / 9.8 # [N]

user_pos = np.array([0.0, 0.0]) # 0=x, 1=y[m]
user_vel = np.array([0.3, 0.0]) # 0=x, 1=y[m/s]
map_size = np.array([10, 10]) # 0=x, 1=y[m]
goal_pos = np.array([10, 5]) # 0=x, 1=y[m]
obstacle_pos = np.array([6, 3]) # 0=x, 1=y[m]

user_pos_lines = user_pos.reshape((2,1))
user_pos_vector = user_vel.reshape((2,1))

Tmax = 0
while True:
    # 引力
    dist_xy = goal_pos - user_pos
    dist_norm = np.linalg.norm(dist_xy, ord=1)
    e = dist_xy / dist_norm
    F_goal = (V0 * e - user_vel) / tau # [N]

    # 斥力
    dist_xy = user_pos - obstacle_pos
    dist_norm = np.linalg.norm(dist_xy, ord=2)
    v_r = dist_xy / dist_norm
    F_obstacle = v_r * U * np.exp((-1) * dist_norm / R)

    # 歩行者の位置計算
    F_all = F_goal + F_obstacle # [N]
    # 加速度 [m/s^2]
    user_acc = F_all / user_mass_N
    # 速度[m/s]
    user_vel += user_acc * delta_sec
    # 位置
    user_pos += user_vel * delta_sec
    Tmax += 1

    # 歩行者とゴールの距離が0.2m以下になったら終了
    if np.linalg.norm(goal_pos - user_pos) < 0.2:
        break
    user_pos_lines = np.append(user_pos_lines, user_pos.reshape((2,1)), axis=1)
    user_pos_vector = np.append(user_pos_vector, user_vel.reshape((2,1)), axis=1)


# 描画
# 障害物
plt.scatter(obstacle_pos[0], obstacle_pos[1], color='b', s=100)
# 軌跡
plt.plot(user_pos_lines[0,:], user_pos_lines[1,:], color='r')
# 軌跡
plt.quiver(user_pos_lines[0,:], user_pos_lines[1,:], user_pos_vector[0,:], user_pos_vector[1,:], color='orange')
# ゴール
plt.scatter(goal_pos[0], goal_pos[1], color='g', s=200)
plt.show()

print("end")


# %%
# アニメーションの描画
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML # type: ignore

# キャンバスを設置
fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot()

ax1.set_xlim(-5, 15)
ax1.set_ylim(-1, 7)

data_pos = user_pos_lines
data_pos_x = data_pos[0]
data_pos_y = data_pos[1]

data_vec = user_pos_vector
data_vec_x = data_vec[0]
data_vec_y = data_vec[1]

#アップデート関数
def update1(i):
    ax1.clear()
        
    ax1.set_xlim(-5, 15)
    ax1.set_ylim(-1, 7)

    data_x = data_pos_x[i]
    data_y = data_pos_y[i]
    vec_x = data_vec_x[i]
    vec_y = data_vec_y[i]

    ax1.scatter(data_x, data_y, color='black')
    ax1.quiver(data_x, data_y, vec_x, vec_y, color='orange')
    ax1.scatter(6, 3, color='red')
    ax1.scatter(10, 5, color='green')
    # 進捗状況を出力する
    progress_rate = (i+1)/Tmax*100
    print("Animation progress={:.3f}%".format(progress_rate), end='\r')
        
#アニメーション作成とgif保存
ani = animation.FuncAnimation(fig, update1, frames=range(Tmax))
    
#グラフ表示
plt.show()

#アニメーションを表示
HTML(ani.to_jshtml())
# %%
