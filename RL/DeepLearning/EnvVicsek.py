
"""
目標：
Vicsek modelにより観測が生成される環境の構築

参考資料：
[1] https://qiita.com/Dolphin7473/items/7a76367a0967e85d376f
[2] Dieter Armbruster, Sébastien Motsch, Andrea Thatcher, Swarming in bounded domains, 2017
[3] https://www.math.umd.edu/~tadmor/ki_net/activities/presentations/4769_686_Hui_Yu_Aachen.pdf

仕様：
エージェント数 N = 50 までの規模で制御とDQNによる学習が可能な環境の構築。
"""

#%%
# ライブラリのインポート

import numpy as np
import torch
import random

class EnvVicsek:
    def __init__(self, num, v0, r, density):
        self.num = num
        self.v0 = v0
        self.r = r
        self.lx = np.sqrt(num / density)
        self.ly = self.lx
        theta_size = (3/16)*np.pi
        # self.add_theta = np.array([-np.pi/2, -(1/3)*np.pi, -(1/6)*np.pi, 0, (1/6)*np.pi, (1/3)*np.pi, np.pi/2])
        self.add_theta = np.array([-theta_size, -(2/3)*theta_size, -(1/3)*theta_size, 0, (1/3)*theta_size, (2/3)*theta_size, theta_size])

    # 周期境界条件
    def boundary_condition(self, x, y, theta):
        # Check if the agent hits the walls
        x = np.where(x < 0, x + self.lx, x)
        x = np.where(x > self.lx, x - self.lx, x)
        y = np.where(y < 0, y + self.ly, y)
        y = np.where(y > self.ly, y - self.ly, y)
        return x, y, theta
    
    # 隣接行列と隣接個体数を計算
    def num_neighbor(self, x, y, theta):
        dx = x[np.newaxis,:] - x[:,np.newaxis]
        dy = y[np.newaxis,:] - y[:,np.newaxis]
        dist = np.sqrt(dx**2 + dy**2)
        neighbor_matrix = dist < self.r
        num_neighbor = np.sum(neighbor_matrix, axis=1)
        return dist, neighbor_matrix, num_neighbor
    
    # 先行研究におけるstate関数
    def arg(self, x, y, theta):
        dx = x[np.newaxis,:] - x[:,np.newaxis] # num_agent x num_agent
        dy = y[np.newaxis,:] - y[:,np.newaxis]
        dis_cul = dx**2 + dy**2 < self.r**2 
        num_neighbor = dis_cul.sum(axis=1)
        # 方向ベクトル
        v = np.array([np.cos(theta), np.sin(theta)]).T
        #print('shape of v={}'.format(v.shape))
        
        cos = np.cos(theta)
        sin = np.sin(theta)
        cos_neighbor = np.dot(dis_cul, cos)
        sin_neighbor = np.dot(dis_cul, sin)
        cos_mean = cos_neighbor / num_neighbor
        sin_mean = sin_neighbor / num_neighbor
        P = np.array([cos_mean, sin_mean]).T
        P_norm = np.linalg.norm(P, axis=1)
        # 正規化近傍平均方向ベクトル
        P_normalize = P / P_norm[:, np.newaxis]
        #print('shape of p_normalize={}'.format(P_normalize.shape))
        
        # vをπ/2回転させたベクトルとPの内積が正の場合はarccos, その他は-arccosを返す
        theta_r = np.pi/2
        rota = np.array([[np.cos(theta_r), -np.sin(theta_r)],[np.sin(theta_r),  np.cos(theta_r)]])
        rota_v = np.tile(rota, (self.num, 1, 1))
        vTop = v.reshape((self.num, 2, 1))
        rota_dot_vTop = np.matmul(rota_v, vTop).reshape((self.num, 2))
        P_dot_vTop = (P * rota_dot_vTop).sum(axis=1)
        
        # 返り値を計算
        naiseki = (P_normalize * v).sum(axis=1)
        arccos = np.arccos(naiseki)
        output = np.where(P_dot_vTop > 0, arccos, -arccos) # 場合分け
        return output

    # agentの状態変数の初期化
    def reset(self):
        x = np.random.uniform(low = 0, high = self.lx, size = self.num)
        y = np.random.uniform(low = 0, high = self.ly, size = self.num)
        # フォン・ミーゼス分布（最初から結構揃っている？）
        #theta = np.random.vonmises(0.0, 4.0, self.num)
        # 一様分布
        theta = np.random.uniform(low=-np.pi, high=np.pi, size=self.num)
        return x, y, theta
    
    # agentの次時刻の状態変数の取得
    def get_StateNext(self, action, x, y, theta):
        #print(f"action={action}")
        action_theta = self.add_theta[action]
        # print(f"action={action_theta}")

        """
        action_cos = np.cos(action_theta)
        action_sin = np.sin(action_theta)
        """
        #print(f"action : cos, sin={action_cos, action_sin}")
        #print(f"native : cos, sin={np.cos(theta), np.sin(theta)}")

        """
        cos_new = np.cos(theta) + action_cos
        sin_new = np.sin(theta) + action_sin
        """

        # 境界における完全弾性衝突
        # cos_new = np.where((x <= 0) | (self.lx <= x), -cos_new, cos_new)
        # sin_new = np.where((y <= 0) | (self.ly <= y), -sin_new, sin_new)
        # theta_new = np.arctan2(np.sin(theta + action_theta), np.cos(theta + action_theta))
        # print(f"theta_new={theta_new}")

        theta_new = theta + action_theta
        theta_new = np.arctan2(np.sin(theta_new), np.cos(theta_new)) #　正規化

        x_new = x + self.v0 * np.cos(theta_new)
        y_new = y + self.v0 * np.sin(theta_new)
        x_new, y_new, theta_new = self.boundary_condition(x_new, y_new, theta_new)
        return x_new, y_new, theta_new
    
    # 当時刻における観測の取得
    def get_obs(self, x, y, theta):
        dx = x[np.newaxis,:] - x[:,np.newaxis]
        dy = y[np.newaxis,:] - y[:,np.newaxis]
        dist = np.sqrt(dx**2 + dy**2)
        neighbor_matrix = dist < self.r
        num_neighbor = np.sum(neighbor_matrix, axis=1)

        dist_neighbor = dist * neighbor_matrix
        dist_mean_neighbor = np.mean(dist_neighbor, axis=1)

        """
        # theta_neighbor = theta * neighbor_matrix
        cos_neighbor = np.cos(theta) * neighbor_matrix
        sin_neighbor = np.sin(theta) * neighbor_matrix
        cos_mean_neighbor = np.mean(cos_neighbor, axis=1)
        sin_mean_neighbor = np.mean(sin_neighbor, axis=1)

        cos = np.cos(theta)
        sin = np.sin(theta)
        norm_mean_neighbor = np.sqrt(cos_mean_neighbor**2 + sin_mean_neighbor**2)
        cos_theta = (cos * cos_mean_neighbor + sin * sin_mean_neighbor) / norm_mean_neighbor
        cos_rota = np.cos(theta + np.pi/2)
        sin_rota = np.sin(theta + np.pi/2)
        cos_theta_rota = (cos_rota * cos_mean_neighbor + sin_rota * sin_mean_neighbor) / norm_mean_neighbor
        """

        # obs1 = np.where(cos_theta_rota > 0, -np.arccos(cos_theta), np.arccos(cos_theta))
        obs1 = self.arg(x, y, theta)
        obs2 = dist_mean_neighbor
        obs3 = num_neighbor

        obs = np.array([obs1, obs2, obs3]).T # new付きから入力時限は１つ
        #print(f"obs={obs}")
        return obs
    
    # 報酬の取得
    def get_reward(self, x, y, theta, x_new, y_new, theta_new):
        dist, neighbor_matrix, num_neigh = self.num_neighbor(x, y, theta)
        _, _, num_neigh_next = self.num_neighbor(x_new, y_new, theta_new)

        """
        cos_neighbor = np.cos(theta) * neighbor_matrix
        sin_neighbor = np.sin(theta) * neighbor_matrix
        order_parameter = np.where(num_neighbor != 0, (np.sqrt(np.sum(cos_neighbor, axis=1)**2 + np.sum(sin_neighbor, axis=1)**2)) / num_neighbor, 0)
        reward = order_parameter # tag = 1
        """

        reward = np.where(num_neigh_next < num_neigh, -1, 0) # tag = 0

        #reward = (2/np.pi)*np.arctan(num_neigh - 1)

        return reward


#%%
# 学習機構の実装

from DQN import DQNAgent, ReplayBuffer, Net
import numpy as np
import torch

def train_dqn(tag, agent, episodes=1000, TimeSteps=250):
    env = EnvVicsek(num, v0, r, density)
    sync_freq = 10  # ターゲットネットワークの同期頻度
    rewards = []

    # 学習
    for episode in range(episodes):
        x, y, theta = env.reset()
        obs = env.get_obs(x, y, theta)
        total_reward = 0

        for t in range(TimeSteps):
            action = []
            for i in range(num):
                action_i = agent[i].get_action(obs[i])
                action.append(action_i)
            action = np.array(action)

            x_next, y_next, theta_next = env.get_StateNext(action, x, y, theta)
            next_obs = env.get_obs(x_next, y_next, theta_next)
            reward = env.get_reward(x, y, theta, x_next, y_next, theta_next)
            #print("reward = {}".format(reward))
            
            for i in range(num):
                done = t == (TimeSteps - 1)
                agent[i].replay_buffer.add(obs[i], action[i], reward[i], next_obs[i], done)
                agent[i].update()
            
            x, y, theta = x_next, y_next, theta_next
            obs = next_obs
            total_reward += np.mean(reward)

            progress_rate = (t+1)/TimeSteps*100
            print("progress={:.3f}%".format(progress_rate), end='\r')
              

        for i in range(num):
            agent[i].update_epsilon()
            if episode % sync_freq == 0:
                agent[i].sync_qnet()

        rewards.append(total_reward)
        print(f"Episode {episode + 1}, Total Reward: {total_reward}")


    # モデルの保存ディレクトリを指定する
    save_dir = 'my_model/'

    # モデルの重みを取得する
    model = [agent[i].qnet.state_dict() for i in range(num)]
            
    # ファイル名の指定
    file_name = 'my_model' + str(tag) + '.pth'

    # 重みを保存する
    torch.save(model, save_dir + file_name)
    print('File name is {}'.format(file_name))

    return rewards


# %%
# 学習の実行
import matplotlib.pyplot as plt

# 学習番号
tag = 1001

# 学習で用いる環境定数
num = 50
v0 = 0.3
r = 0.75
density = 2.0

action_size = 7
state_size = 3
policy = "epsilon_greedy"

if __name__ == "__main__":
    agents = [DQNAgent(state_size, action_size, policy) for _ in range(num)]
    trained_rewards = train_dqn(tag, agents)

    plt.plot(trained_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Rewards')
    plt.title(f'Training Rewards\nnum={num}, v0={v0}, r={r}, density={density}, (state, action)={state_size, action_size}')
    fig_name = f"fig/training_rewards_{tag}.png"
    plt.savefig(fig_name)
    plt.show()
    


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
        self.dt = 0.5

    # ノイズ項
    def generator(self):
        return self.noise * np.random.uniform(high=1.0, low=-1.0, size=self.num)
    
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

        theta_neighbor = theta * neighbor_matrix
        cos_neighbor = np.cos(theta_neighbor)
        sin_neighbor = np.sin(theta_neighbor)
        cos_mean_neighbor = np.mean(cos_neighbor, axis=1)
        sin_mean_neighbor = np.mean(sin_neighbor, axis=1)
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

        # 時刻 t におけるagentの向きの平均を計算
        theta_neighbor = theta * neighbor_matrix
        cos_neighbor = np.cos(theta_neighbor)
        sin_neighbor = np.sin(theta_neighbor)
        cos_mean_neighbor = np.mean(cos_neighbor, axis=1)
        sin_mean_neighbor = np.mean(sin_neighbor, axis=1)
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
    
    # シミュレーションの実行
    def simulate(self, Tmax):
        x, y, theta = self.reset()
        state_record = []
        t = 0
        num_steps = int(Tmax / self.dt)

        for t in range(num_steps):
            order_parameter = self.get_order(x, y, theta)

            state = [x, y, theta, order_parameter]
            state_record.append(state)

            x_next, y_next, theta_next = self.step_inBD(x, y, theta)
            x, y, theta = x_next, y_next, theta_next

            progress_rate = (t+1)/num_steps*100
            print("progress={:.3f}%".format(progress_rate), end='\r')
            t += 1

        return state_record
    
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
            for xi, yi in zip(x, y):
                circle = Circle((xi, yi), self.r, color='blue', fill=False, linestyle='dotted')
                ax.add_patch(circle)
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

num = 50
v0 = 0.3
r = 0.75
density = 2.0
noise = 0.1
simulator = EnvVicsek_test(num, v0, r, density, noise)

Tmax = 50
state_record = simulator.simulate(Tmax)
simulator.animate(state_record)


# %%
# 学習済みagentによるシミュレーション

from DQN import DQNAgent, ReplayBuffer, Net
import numpy as np
import torch

def simulate(model, Tmax):
    # 定数
    num = 1
    v0 = 0.5
    r = 0.75
    density = 2.0
    noise = 0.0
    action_size = 7
    state_size = 3
    policy = "greedy"

    # メソッドの呼び出し
    agent = [DQNAgent(state_size, action_size, policy) for _ in range(num)]
    env = EnvVicsek(num, v0, r, density)

    # 各agentに学習済みNNをロード
    for i in range(num):
        agent[i].load_state_dict(model[i])

    x, y, theta = env.reset()
    obs = env.get_obs(x, y, theta)
    
    state_record = []
    reward_record = []

    for t in range(Tmax):
        state = [x, y, theta]
        state_record.append(state)

        action = []
        for i in range(num):
            action_i = agent[i].get_action(obs[i])
            action.append(action_i)
        action = np.array(action)

        x_next, y_next, theta_next = env.get_StateNext(action, x, y, theta)
        next_obs = env.get_obs(x_next, y_next, theta_next)
        reward = env.get_reward(x, y, theta, x_next, y_next, theta_next)
                
        x, y, theta = x_next, y_next, theta_next
        obs = next_obs
        
        reward_record.append(reward)

        # 進捗状況を出力する
        progress_rate = (t+1)/Tmax*100
        print("progress={:.3f}%".format(progress_rate), end='\r')
    
    return state_record, reward_record 


#%%
# 学習済みエージェントによるシミュレーションを実行

import numpy as np
import torch

# 学習済みNNを呼び出し
model_tag = 1001
model_name = f"my_model/my_model{model_tag}.pth"
model = torch.load(model_name)

# シミュレーション
state_record, reward_record = simulate(model, Tmax=100)


# %%
# テスト結果のアニメーション表示

import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Circle
from IPython.display import HTML

def animate(state_record, reward_record, model_tag, save):
    # 定数
    num = 1
    v0 = 0.5
    r = 0.75
    density = 2.0
    noise = 0.0
    action_size = 7
    state_size = 3
    lx = ly = np.sqrt(num / density)

    # キャンバスを設置
    video_length = len(state_record)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot()

    ax.set_xlim(0, lx)
    ax.set_ylim(0, ly)

    #アップデート関数
    def update(i):
        ax.clear()
        ax.set_xlim(0, lx)
        ax.set_ylim(0, ly)

        data = state_record[i]
        x, y, theta = data[0], data[1], data[2]
        dx = x[np.newaxis,:] - x[:,np.newaxis]
        dy = y[np.newaxis,:] - y[:,np.newaxis]
        neighbor = np.sqrt(dx**2 + dy**2) < r
        num_neighbor = np.sum(neighbor, axis=1)
        order = (1/num_neighbor) * np.sqrt(np.sum(np.cos(theta)*neighbor, axis=1)**2 + np.sum(np.sin(theta)*neighbor, axis=1)**2)

        ax.quiver(x, y, np.cos(theta), np.sin(theta), color='black')
        ax.set_title('N={}, v0={}, R={}, density={}, noise={} ,time={:.2f}, order parameter={:.2f}'.format(num, v0, r, density, noise, (i+1), np.mean(order)), fontsize=14)
        for xi, yi in zip(x, y):
            circle = Circle((xi, yi), r, color='blue', fill=False, linestyle='dotted')
            ax.add_patch(circle)
        # 進捗状況を出力する
        progress_rate = (i+1)/video_length*100
        print("Animation progress={:.3f}%".format(progress_rate), end='\r')
                
    #アニメーション作成
    ani = animation.FuncAnimation(fig, update, frames=range(video_length))

    # アニメーションの保存
    if save == "yes":
        print("Aniation is being saved.")
        animation_tag = f"animation/model{model_tag}.mp4"
        ani.save(animation_tag, writer='ffmpeg')

    #グラフ表示
    plt.show()

    #アニメーションを表示
    return HTML(ani.to_jshtml())


#%%
# アニメーションの作成
save = "no"
animate(state_record, reward_record, model_tag, save)
# %%
import numpy as np
import random

a = np.arange(3) + 2
index = np.random.randint(low=0, high=3, size=10)
print(f"a={a}, index={index}")

b = a[index]
print(f"b={b}")

# %%
