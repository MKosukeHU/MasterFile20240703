"""
目標：
Vicsek modelの実装。VAE学習用のデータセット生成。

"""


#%%
# 構築環境下でのagentの挙動を確認。
# 行動空間は離散だが、本環境ではVicsek modelを採用。

import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Circle
from IPython.display import HTML

class EnvVicsek_test:
    def __init__(self, num, v0, r, density, noise):
        self.num = num
        self.v0 = v0
        self.r = r
        self.lx = np.sqrt(num / density)
        self.ly = self.lx
        self.noise = noise
        self.dt = 1.0

    # ノイズ項
    def generator(self):
        return self.noise * np.random.uniform(high=1.0, low=-1.0, size=self.num)
    
    # 隣接行列などの計算　
    def cul_neighbor(self, x, y, theta):
        dx = x[np.newaxis,:] - x[:,np.newaxis]
        dy = y[np.newaxis,:] - y[:,np.newaxis]
        dist = np.sqrt(dx**2 + dy**2)
        neighbor_matrix = dist < self.r
        num_neighbor = np.sum(neighbor_matrix, axis=1)
        return dx, dy, dist, neighbor_matrix, num_neighbor
    
    # 秩序変数の計算
    def get_order(self, x, y, theta):
        dx = x[np.newaxis,:] - x[:,np.newaxis]
        dy = y[np.newaxis,:] - y[:,np.newaxis]
        dist = np.sqrt(dx**2 + dy**2)
        neighbor_matrix = dist < self.r

        num_neighbor = np.sum(neighbor_matrix, axis=1)
        cos_neighbor = np.cos(theta) * neighbor_matrix
        sin_neighbor = np.sin(theta) * neighbor_matrix
        order_parameter = np.where(num_neighbor != 0, (np.sqrt(np.sum(cos_neighbor, axis=1)**2 + np.sum(sin_neighbor, axis=1)**2)) / num_neighbor, 0)
        return order_parameter
    
    # 状態の計算
    def arg(self, x, y, theta):
        dx = x[:, np.newaxis] - x[np.newaxis, :]  # num_agent x num_agent
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dis_cul = (dx**2 + dy**2) < self.r**2
        num_neighbor = dis_cul.sum(axis=1)
        
        # 方向ベクトル
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        v = np.stack((cos_theta, sin_theta), axis=-1)
        
        cos_neighbor = np.dot(dis_cul, cos_theta)
        sin_neighbor = np.dot(dis_cul, sin_theta)
        cos_mean = cos_neighbor / num_neighbor
        sin_mean = sin_neighbor / num_neighbor
        P = np.stack((cos_mean, sin_mean), axis=-1)
        
        P_norm = np.linalg.norm(P, axis=1, keepdims=True)
        P_normalize = P / P_norm
        
        # vをπ/2回転させたベクトルとPの内積
        rota_v = np.stack((-sin_theta, cos_theta), axis=-1)
        P_dot_vTop = np.einsum('ij,ij->i', P, rota_v)
        
        # 返り値を計算
        naiseki = np.einsum('ij,ij->i', P_normalize, v)
        arccos = np.arccos(np.clip(naiseki, -1.0, 1.0))
        output = np.where(P_dot_vTop > 0, arccos, -arccos)
        return output

    # 周期境界条件
    def boundary_condition(self, x, y, theta):
        # periodic boundary condition, 周期境界条件
        x = np.where(x < 0, x + self.lx, x)
        x = np.where(x > self.lx, x - self.lx, x)
        y = np.where(y < 0, y + self.ly, y)
        y = np.where(y > self.ly, y - self.ly, y)
        return x, y, theta

    # agentの状態変数の初期化
    def reset(self):
        x = np.random.uniform(low = 0, high = self.lx, size = self.num)
        y = np.random.uniform(low = 0, high = self.ly, size = self.num)
        theta = np.random.vonmises(0.0, 4.0, self.num)
        return x, y, theta
    
    # agentの次時刻の状態変数の取得
    def step_pbc(self, x, y, theta):
        dx = x[np.newaxis,:] - x[:,np.newaxis]
        dy = y[np.newaxis,:] - y[:,np.newaxis]
        dist = np.sqrt(dx**2 + dy**2)
        neighbor_matrix = dist < self.r
        num_neighbor = np.sum(neighbor_matrix, axis=1)

        cos_neighbor = np.cos(theta) * neighbor_matrix
        sin_neighbor = np.sin(theta) * neighbor_matrix
        cos_mean_neighbor = np.sum(cos_neighbor, axis=1) / num_neighbor
        sin_mean_neighbor = np.sum(sin_neighbor, axis=1) / num_neighbor
        noize_term = self.generator()
        theta_new = np.arctan2(sin_mean_neighbor, cos_mean_neighbor) + noize_term

        x_new = x + self.v0 * np.cos(theta_new) * self.dt
        y_new = y + self.v0 * np.sin(theta_new) * self.dt
        x_new, y_new, theta_new = self.boundary_condition(x_new, y_new, theta_new)
        return x_new, y_new, theta_new
    
    # 閉空間におけるagentの更新式, 参考：[2], [3]
    def step_inBD(self, x, y, theta):
        dx = x[np.newaxis,:] - x[:,np.newaxis]
        dy = y[np.newaxis,:] - y[:,np.newaxis]
        dist = np.sqrt(dx**2 + dy**2)
        neighbor_matrix = dist < self.r
        num_neighbor = np.sum(neighbor_matrix, axis=1)

        # 時刻 t におけるagentの向きの平均を計算
        cos_neighbor = np.cos(theta) * neighbor_matrix
        sin_neighbor = np.sin(theta) * neighbor_matrix
        cos_mean_neighbor = np.sum(cos_neighbor, axis=1) / num_neighbor
        sin_mean_neighbor = np.sum(sin_neighbor, axis=1) / num_neighbor
        noise_term = self.generator()
        theta_mean = np.arctan2(sin_mean_neighbor, cos_mean_neighbor)

        # 境界反射の計算と運動方向の更新
        cos = np.cos(theta)
        sin = np.sin(theta)
        cos_new = np.where((x <= 0) | (self.lx <= x), -cos, np.cos(theta))
        sin_new = np.where((y <= 0) | (self.ly <= y), -sin, np.sin(theta))
        theta_new = np.arctan2(sin_new, cos_new) + noise_term

        # 時刻 t + dt におけるagentの位置の計算
        x_new = x + self.v0 * np.cos(theta_new) * self.dt
        y_new = y + self.v0 * np.sin(theta_new) * self.dt
        return x_new, y_new, theta_new
    
    # 学習用データの生成
    def gen_data(self, x, y, theta):
        # data = self.arg(x, y, theta), data set No1    

        # data set No2では0番目エージェントの情報のみを使用する
        dx, dy, dist, neighbor_matrix, num_neighbor = self.cul_neighbor(x, y, theta)
        invers_num = 1 / num_neighbor
        dx_neighbor_mean = invers_num * np.sum(dx * neighbor_matrix, axis=1)
        dy_neighbor_mean = invers_num * np.sum(dy * neighbor_matrix, axis=1)
        data = np.array([invers_num[0], dx_neighbor_mean[0], dy_neighbor_mean[0]])

        return data
    
    # シミュレーションの実行
    def simulate(self, Tmax):
        x, y, theta = self.reset()
        state_record = []
        data_record = []
        t = 0
        num_steps = int(Tmax / self.dt)

        for t in range(num_steps):
            order_parameter = self.get_order(x, y, theta)

            data = self.gen_data(x, y, theta)
            data_record.append(data)

            state = [x, y, theta, order_parameter]
            state_record.append(state)

            x_next, y_next, theta_next = self.step_pbc(x, y, theta)
            x, y, theta = x_next, y_next, theta_next

            progress_rate = (t+1)/num_steps*100
            print("progress={:.3f}%".format(progress_rate), end='\r')
            t += 1

        return state_record,  data_record
    
    # テスト結果のアニメーション表示
    def animate(self, state_record):
        # キャンバスを設置
        video_length = len(state_record)
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot()

        ax.set_xlim(0, self.lx)
        ax.set_ylim(0, self.ly)

        #アップデート関数
        def update(i):
            ax.clear()
            ax.set_xlim(0, self.lx)
            ax.set_ylim(0, self.ly)

            data = state_record[i]
            x, y, theta, order = data[0], data[1], data[2], data[3]

            ax.quiver(x, y, np.cos(theta), np.sin(theta), color='black')
            ax.set_title('N={}, v0={}, R={}, density={}, noise={} ,time={:.2f}, order parameter={:.2f}'.format(self.num, self.v0, self.r, density, noise, (i+1)*self.dt, np.mean(order)), fontsize=14)
            
            """
            for xi, yi in zip(x, y):
                circle = Circle((xi, yi), self.r, color='blue', fill=False, linestyle='dotted')
                ax.add_patch(circle)
            """

            # 進捗状況を出力する
            progress_rate = (i+1)/video_length*100
            print("Animation progress={:.3f}%".format(progress_rate), end='\r')
                
        #アニメーション作成とgif保存
        ani = animation.FuncAnimation(fig, update, frames=range(video_length))
            
        #グラフ表示
        plt.show()

        #アニメーションを表示
        return HTML(ani.to_jshtml())
        

#%%
# テストの実行

num = 100
v0 = 0.3
r = 1.0
density = 3.0
noise = 0.5
simulator = EnvVicsek_test(num, v0, r, density, noise)

Tmax = 200
state_record, data_record = simulator.simulate(Tmax)
simulator.animate(state_record)


# %%

print(f"data=(1 / num neighbor, dx neighbor, dy neighbor)\n{np.array(data_record)}")

# %%
import numpy as np
import os

# 保存先ディレクトリとファイル名
tag = 2
dir_path = "../../VAE/TestData"
file_name = f"data{tag}.txt"
file_path = os.path.join(dir_path, file_name)

# ディレクトリが存在するか確認し、存在しない場合は作成
os.makedirs(dir_path, exist_ok=True)

# データを保存
np.savetxt(file_path, np.array(data_record))
# %%
