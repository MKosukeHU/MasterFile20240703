
"""
目標：
1対1の Prey-Predator model を使って、強化学習を行う。状態は角度情報の一次元とする。

参考資料：
[1] 卒研2024
[2] https://www.jstage.jst.go.jp/article/sci/SCI06/0/SCI06_0_32/_pdf/-char/ja, RLスキームによる群れ行動モデルの検討
[3] https://www.jstage.jst.go.jp/article/jacc/49/0/49_0_195/_pdf, RLによる群行動発言のモデリング
"""


#%%
# Pray-Pradation modelクラスの実装

# ライブラリ
import numpy as np
import random

# クラスの実装
class Prey_Predator:
    def __init__(self, v_prey, v_predator, lx, ly, noise):
        self.v_prey = v_prey
        self.v_predator = v_predator
        self.lx = lx
        self.ly = ly
        self.noise = noise
        self.max = (3/16)*np.pi
        self.action_set = np.array([-self.max, -(2/3)*self.max, -(1/3)*self.max, 0, (1/3)*self.max, (2/3)*self.max, self.max])

    # 周期境界条件
    def PBC(self, x, y):
        # periodic boundary condition, 周期境界条件
        x = np.where(x < 0, x + self.lx, x)
        x = np.where(x > self.lx, x - self.lx, x)
        y = np.where(y < 0, y + self.ly, y)
        y = np.where(y > self.ly, y - self.ly, y)
        return x, y
    
    # 境界面完全弾性衝突条件
    def BD(self, x, y, theta):
        # 境界反射の計算と運動方向の更新
        cos = np.cos(theta)
        sin = np.sin(theta)
        cos_next = np.where((x <= 0) | (self.lx <= x), -cos, np.cos(theta))
        sin_next = np.where((y <= 0) | (self.ly <= y), -sin, np.sin(theta))
        theta_next = np.arctan2(sin_next, cos_next)
        return theta_next
    
    # ノイズ生成器
    def generator(self):
        return np.random.uniform(low=-1.0, high=1.0)

    # 全座標のリセット
    def reset(self):
        x_prey = np.random.uniform(low=0, high=self.lx/2)
        y_prey = np.random.uniform(low=0, high=self.ly/2)
        theta_prey = np.random.vonmises(mu=0, kappa=0.4)
        x_predator = np.random.uniform(low=self.lx/2, high=self.lx)
        y_predator = np.random.uniform(low=self.ly/2, high=self.ly)
        theta_predator = np.random.vonmises(mu=0, kappa=0.4)
        return x_prey, y_prey, theta_prey, x_predator, y_predator, theta_predator
    
    # Predatorの状態変数の生成
    def get_state_prey(self, x_prey, y_prey, theta_prey, x_predator, y_predator, theta_predator):
        dx = x_prey - x_predator
        dy = y_prey - y_predator
        dist = np.sqrt(dx**2 + dy**2)
        vx = np.cos(theta_predator)
        vy = np.sin(theta_predator)
        cos_psi = (dx * vx + dy * vy) / dist

        vx_rot = np.cos(theta_predator + np.pi/2)
        vy_rot = np.sin(theta_predator + np.pi/2)
        cos_psi_rot = (dx * vx_rot + dy * vy_rot) / dist

        psi = np.where(0 < cos_psi_rot, -np.arccos(cos_psi), np.arccos(cos_psi))
        return psi

    # Predatorを1ステップ進める
    def step_predator(self, x, y, theta, action):
        theta_action = self.action_set[action]
        noise_term = self.noise * self.generator()
        theta_next = np.arctan2(np.sin(theta + theta_action), np.cos(theta + theta_action)) + noise_term
        x_next = x + self.v_predator * np.cos(theta_next)
        y_next = y + self.v_predator * np.sin(theta_next)
        x_next, y_next = self.PBC(x_next, y_next)
        return x_next, y_next, theta_next
    
    # Preyを1ステップ進める
    def step_prey(self, x, y, theta):
        theta_next = theta + self.max * np.random.rand()
        x_next = x + self.v_prey * np.cos(theta_next) # x方向への移動。cos(θ)
        y_next = y + self.v_prey * np.sin(theta_next) # y方向への移動。sin(θ)
        x_next, y_next = self.PBC(x_next, y_next)
        return x_next, y_next, theta_next
    
    # 報酬関数
    def get_reward(self, x_prey, y_prey, x_predator, y_predator, x_prey_next, y_prey_next, x_predator_next, y_predator_next):
        dx = x_prey - x_predator
        dy = y_prey - y_predator
        dist = np.sqrt(dx**2 + dy**2)

        dx_next = x_prey_next - x_predator_next
        dy_next = y_prey_next - y_predator_next
        dist_next = np.sqrt(dx_next**2 + dy_next**2)
        reward = np.where(dist < dist_next, 1.0, 0.0)
        return reward


#%%
# 学習関数の実装 # クラス化してアニメーション機能を実装しましょう。

# ライブラリ
import numpy as np
from Q_lerarning import QTableAgent, QTableAgent_min
from matplotlib import animation
from IPython.display import HTML
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

class Learning_and_simualtion:
    def __init__(self, num, v_prey, v_predator, lx, ly, noise, state_size, action_size):
        self.num = num
        self.v_prey = v_prey
        self.v_predator = v_predator
        self.lx = lx
        self.ly = ly
        self.noise = noise
        self.state_size = state_size
        self.action_size = action_size
        self.policy_train = "epsilon_greedy"
        self.policy_simulation = "greedy"
        
    # 学習
    def Run(self, episodes, time):
        agent = QTableAgent_min(self.state_size, self.action_size, self.num, self.policy_train, episodes)
        env = Prey_Predator(self.v_prey, self.v_predator, self.lx, self.ly, self.noise)
        reward_record = []

        for episode in range(episodes):
            tortal_reward = 0

            x_prey, y_prey, theta_prey, x_predator, y_predator, theta_predator = env.reset()
            for t in range(time):
                state = env.get_state_prey(x_prey, y_prey, theta_prey, x_predator, y_predator, theta_predator)
                action = agent.get_action(state, episode)

                x_prey_next, y_prey_next, theta_prey_next = env.step_prey(x_prey, y_prey, theta_prey)
                x_predator_next, y_predator_next, theta_predator_next = env.step_predator(x_predator, y_predator, theta_predator, action)
                state_next = env.get_state_prey(x_prey_next, y_prey_next, theta_prey_next, x_predator_next, y_predator_next, theta_predator_next)

                done = True if t == time else False
                reward = env.get_reward(x_prey, y_prey, x_predator, y_predator, x_prey_next, y_prey_next, x_predator_next, y_predator_next)
                tortal_reward += reward.mean()

                agent.update(state, action, reward, state_next, done)

                x_prey, y_prey, theta_prey = x_prey_next, y_prey_next, theta_prey_next
                x_predator, y_predator, theta_predator = x_predator_next, y_predator_next, theta_predator_next

            print(f"Episode : {episode + 1} / {episodes} Tortal Reward : {tortal_reward}")
            reward_record.append(tortal_reward)

        Qtable = agent.get_q_table()
        return reward_record, Qtable
    
    # シミュレーション
    def simulate(self, time, file_path):
        # インスタンス化
        agent = QTableAgent_min(self.state_size, self.action_size, self.num, self.policy_simulation, max_episodes=0)
        env = Prey_Predator(self.v_prey, self.v_predator, self.lx, self.ly, self.noise)

        # エージェントにQテーブルをダウンロード
        agent.load_q_table(file_path)

        prey_record = []
        predator_record = []
        reward_record = []

        x_prey, y_prey, theta_prey, x_predator, y_predator, theta_predator = env.reset()
        for t in range(time):
            prey = np.array([x_prey, y_prey, theta_prey])
            prey_record.append(prey)
            predator = np.array([x_predator, y_predator, theta_predator])
            predator_record.append(predator)

            state = env.get_state_prey(x_prey, y_prey, theta_prey, x_predator, y_predator, theta_predator)
            action = agent.get_action(state, episode=0)

            x_prey_next, y_prey_next, theta_prey_next = env.step_prey(x_prey, y_prey, theta_prey)
            x_predator_next, y_predator_next, theta_predator_next = env.step_predator(x_predator, y_predator, theta_predator, action)
    
            reward = env.get_reward(x_prey, y_prey, x_predator, y_predator, x_prey_next, y_prey_next, x_predator_next, y_predator_next)
            reward_record.append(np.mean(reward))

            x_prey, y_prey, theta_prey = x_prey_next, y_prey_next, theta_prey_next
            x_predator, y_predator, theta_predator = x_predator_next, y_predator_next, theta_predator_next
        return prey_record, predator_record, reward_record
    
    # アニメーションの作成
    def animate(self, time, file_path):
        # シミュレーション
        prey_record, predator_record, reward_record = self.simulate(time, file_path)

        # キャンバスを設置
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot()

        lx_sim = self.lx + 5
        ly_sim = self.ly + 5
        ax.set_xlim(-5, lx_sim)
        ax.set_ylim(-5, ly_sim)

        #アップデート関数
        def update(i):
            ax.clear()
            ax.set_xlim(-5, lx_sim)
            ax.set_ylim(-5, ly_sim)
            ax.axvspan(xmin=-5, xmax=0, color="yellow", alpha=0.1)
            ax.axvspan(xmin=self.lx, xmax=lx_sim, color="yellow", alpha=0.1)
            ax.axhspan(ymin=-5, ymax=0, xmin=0, xmax=self.lx, color="yellow", alpha=0.1)
            ax.axhspan(ymin=self.ly, ymax=ly_sim, xmin=0, xmax=self.lx, color="yellow", alpha=0.1)

            prey = prey_record[i]
            predator = predator_record[i]

            ax.quiver(prey[0], prey[1], np.cos(prey[2]), np.sin(prey[2]), color='blue')
            ax.quiver(predator[0], predator[1], np.cos(predator[2]), np.sin(predator[2]), color='red')
            ax.set_title(f"Model : {file_path}, Time : {i+1}, Reward : {reward_record[i]:.3f}"
                         + f"\nPrey (Model Control) : Blue, v : {v_prey}"
                         + f"\nPredator (Q-learning) : Red, v : {v_predator}")

            # 進捗状況を出力する
            progress_rate = (i+1)/time*100
            print("Animation progress={:.3f}%".format(progress_rate), end='\r')
                
        #アニメーション作成とgif保存
        ani = animation.FuncAnimation(fig, update, frames=range(time))

        #アニメーションを表示
        return HTML(ani.to_jshtml())
    
    # グラフの描画
    def Plot_graph(self, reward_record, model_path):
        # 描画
        plt.plot(np.array(reward_record) / time)
        plt.ylim(0, 1)
        plt.title(rf"Model : {model_path}, " 
                  + rf"Reward : $\frac{{1}}{{\| \mathbf{{r}}_{{Prey}} - \mathbf{{r}}_{{Predator}} \|}}$")
        plt.xlabel(f"Episode, (Episode : {episodes})")
        plt.ylabel(f"Tortal Reward / Time, (Time : {time})")
        return plt.show()
    
    # Qtableのヒートマップ描画
    def Map_Qtable(self, Qtabel, model_path):
        im = plt.imshow(Qtabel[0], cmap='jet', origin='lower', aspect='auto')
        plt.title("Heatmap of Qtabel, Model : " + model_path)
        plt.xlabel("Action")
        plt.ylabel("State")
        plt.colorbar(im)
        return plt.show()
    
    # 学習データの保存
    def SaveData(self, reward, Qtabel, model_path):
        # rewardの保存
        dir_reward = "RewardRecord/"
        np.save(dir_reward + model_path, reward)

        # Qtableの保存
        dir_Qtable = "QtableRcord"
        np.save(dir_Qtable + model_path, Qtabel)

        # 学習用パラメータの保存
        parameter = np.array([self.num, self.v_prey, self.v_predator, self.lx, self.ly, self.noise, self.state_size, self.action_size])
        dir_parameter = "ParameterRecord/"
        np.savetxt(dir_parameter + model_path, parameter)
        return print("Data saving completed")


# %%
# 学習の実行

# ライブラリ
import numpy as np
import random
from Q_lerarning import QTableAgent

# 定数
num = 1
v_prey = 0.5
v_predator = 1.0
lx = ly = 20
noise = 0.0

state_size = 32
action_size = 7

# 学習時間の設定
episodes = 1000
time = 10000

machine = Learning_and_simualtion(num, v_prey, v_predator, lx, ly, noise, state_size, action_size)

# 学習の実行とQテーブル等の保存
tag = 1
model_path = f"model{tag}.npy"
reward_record, Qtable = machine.Run(episodes, time)
machine.SaveData(reward_record, Qtable, model_path)
machine.Plot_graph(reward_record, model_path)
machine.Map_Qtable(Qtable, model_path)


# %%
# エージェントのアニメーションを作成

model_tag = 1
time_simulation = 100
machine.animate(time=time_simulation, file_path=f"QtableRecord/model{model_tag}.npy")

#%%
