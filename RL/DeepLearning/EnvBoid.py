
"""
Boid modelベースのマルチエージェント環境

目標：
行動がBoid modelに準拠している環境設定の構築。

参考資料：
[1] https://mas.kke.co.jp/model/boid-model/
[2] https://www.jstage.jst.go.jp/article/sicejl/52/3/52_234/_pdf
"""


#%%
import numpy as np
import torch
import random

class EnvBoid:
    def __init__(self, num, v0, r, density):
        self.num = num
        self.v0 = v0
        self.r = r
        self.lx = np.sqrt(num / density)
        self.ly = self.lx
        
    # 周期境界条件
    def boundary_condition(self, x, y, theta):
        x = np.where(x < 0, x + self.lx, x)
        x = np.where(x > self.lx, x - self.lx, x)
        y = np.where(y < 0, y + self.ly, y)
        y = np.where(y > self.ly, y - self.ly, y)
        return x, y, theta
    
    # 隣接行列と隣接個体数を計算
    def cul_neighbor(self, x, y, theta):
        dx = x[np.newaxis,:] - x[:,np.newaxis]
        dy = y[np.newaxis,:] - y[:,np.newaxis]
        dist = np.sqrt(dx**2 + dy**2)
        neighbor_matrix = dist < self.r
        num_neighbor = np.sum(neighbor_matrix, axis=1)
        return dx, dy, dist, neighbor_matrix, num_neighbor
    
    # 先行研究におけるstate関数
    def arg(self, x, y, theta):
        dx = x[np.newaxis,:] - x[:,np.newaxis]
        dy = y[np.newaxis,:] - y[:,np.newaxis]
        dis_cul = dx**2 + dy**2 < self.r**2 
        num_neighbor = dis_cul.sum(axis=1)
        
        v = np.array([np.cos(theta), np.sin(theta)]).T
        
        cos = np.cos(theta)
        sin = np.sin(theta)
        cos_neighbor = np.dot(dis_cul, cos)
        sin_neighbor = np.dot(dis_cul, sin)
        cos_mean = cos_neighbor / num_neighbor
        sin_mean = sin_neighbor / num_neighbor
        P = np.array([cos_mean, sin_mean]).T
        P_norm = np.linalg.norm(P, axis=1)
        P_normalize = P / P_norm[:, np.newaxis]
        
        theta_r = np.pi / 2
        rota = np.array([[np.cos(theta_r), -np.sin(theta_r)], [np.sin(theta_r), np.cos(theta_r)]])
        rota_v = np.tile(rota, (self.num, 1, 1))
        vTop = v.reshape((self.num, 2, 1))
        rota_dot_vTop = np.matmul(rota_v, vTop).reshape((self.num, 2))
        P_dot_vTop = (P * rota_dot_vTop).sum(axis=1)
        
        naiseki = (P_normalize * v).sum(axis=1)
        arccos = np.arccos(naiseki)
        output = np.where(P_dot_vTop > 0, arccos, -arccos)
        return output

    # agentの状態変数の初期化
    def reset(self):
        x = np.random.uniform(low=0, high=self.lx, size=self.num)
        y = np.random.uniform(low=0, high=self.ly, size=self.num)
        theta = np.random.uniform(low=-np.pi, high=np.pi, size=self.num)
        return x, y, theta
    
    # 行動の計算
    def generate_action(self, x, y, theta, action):
        dx, dy, dist, neighbor_matrix, num_neighbor = self.cul_neighbor(x, y, theta)
        dx_neighbor = dx * neighbor_matrix
        dy_neighbor = dy * neighbor_matrix
        dist_neighbor = dist * neighbor_matrix
        inverse_num = 1 / num_neighbor
        vx = np.cos(theta)
        vy = np.sin(theta)
        vx_neighbor = vx * neighbor_matrix
        vy_neighbor = vy * neighbor_matrix
        
        # 結合力
        dx_mean = inverse_num * np.sum(dx_neighbor, axis=1)
        dy_mean = inverse_num * np.sum(dy_neighbor, axis=1)
        theta_coh = np.arctan2(dy_mean, dx_mean)

        # 分離力
        theta_div = np.arctan2(-dy_mean, -dx_mean)

        # 整列力
        cos_mean = inverse_num * np.sum(vx_neighbor, axis=1)
        sin_mean = inverse_num * np.sum(vy_neighbor, axis=1)
        theta_ali = np.arctan2(sin_mean, cos_mean)

        # 角度の選出
        theta_list = np.array([theta_coh, theta_div, theta_ali, theta]).T
        agent_index = np.arange(self.num)
        theta_new = theta_list[agent_index, action]

        # 反映
        x_new = x + self.v0 * np.cos(theta_new)
        y_new = y + self.v0 * np.sin(theta_new)
        x_new, y_new, theta_new = self.boundary_condition(x_new, y_new, theta_new)
        return x_new, y_new, theta_new
        
    # agentの次時刻の状態変数の取得
    def get_StateNext(self, action, x, y, theta):
        x_new, y_new, theta_new = self.generate_action(x, y, theta, action)
        x_new, y_new, theta_new = self.boundary_condition(x_new, y_new, theta_new)
        return x_new, y_new, theta_new
    
    # 当時刻における観測の取得
    def get_obs(self, x, y, theta):
        dx, dy, dist, neighbor_matrix, num_neighbor = self.cul_neighbor(x, y, theta)
        dist_neighbor = dist * neighbor_matrix
        dist_mean_neighbor = (1 / num_neighbor) * np.sum(dist_neighbor, axis=1)
        # dx_mean_neighbor = (1 / num_neighbor) * np.sum(dx * neighbor_matrix, axis=1)
        # dy_mean_neighbor = (1 / num_neighbor) * np.sum(dy * neighbor_matrix, axis=1)

        obs1 = self.arg(x, y, theta)
        obs2 = dist_mean_neighbor
        obs3 = num_neighbor

        obs = np.array([obs1, obs2, obs3]).T
        return obs
    
    # 報酬の取得
    def get_reward(self, x, y, theta, x_new, y_new, theta_new):
        dist, _, _, _, num_neighbor = self.cul_neighbor(x, y, theta)
        # dist_next, _, _, _, num_neigh_next = self.cul_neighbor(x_new, y_new, theta_new)
        # reward = np.where(num_neigh_next > num_neigh, 1, 0)

        reward = np.where(num_neighbor == 1, 1, 0)
        return reward
    

#%%
# 学習機構の実装

from DQN import DQNAgent, ReplayBuffer, Net
import numpy as np
import torch

def train_dqn(tag, agent, episodes=500, TimeSteps=250):
    env = EnvBoid(num, v0, r, density)
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
    file_name = 'Boid_my_model' + str(tag) + '.pth'

    # 重みを保存する
    torch.save(model, save_dir + file_name)
    print('File name is {}'.format(file_name))

    return rewards


#%%
# 学習の実行
import matplotlib.pyplot as plt

# 学習番号
tag = 1

# 学習で用いる環境定数
num = 50
v0 = 0.5
r = 1.0
density = 0.5

action_size = 4
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
    fig_name = f"fig/EnvBoid_train_rewards_{tag}.png"
    plt.savefig(fig_name)
    plt.show()

#%%
# 学習済みagentによるシミュレーション

from DQN import DQNAgent, ReplayBuffer, Net
import numpy as np
import torch

def simulate(model, Tmax):
    # 定数
    num = 50
    v0 = 0.5
    r = 1.0
    density = 1.0
    noise = 0.0
    action_size = 4
    state_size = 3
    policy = "greedy"

    # メソッドの呼び出し
    agent = [DQNAgent(state_size, action_size, policy) for _ in range(num)]
    env = EnvBoid(num, v0, r, density)

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
model_tag = 1
model_name = f"my_model/Boid_my_model{model_tag}.pth"
model = torch.load(model_name)

# シミュレーション
state_record, reward_record = simulate(model, Tmax=100)


#%%
# テスト結果のアニメーション表示

import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Circle
from IPython.display import HTML

def animate(state_record, reward_record, model_tag, save):
    # 定数
    num = 50
    v0 = 0.5
    r =  1.0
    density = 1.0
    noise = 0.0
    action_size = 4
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
save = "yes"
animate(state_record, reward_record, model_tag, save)


# %%
