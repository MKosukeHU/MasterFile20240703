
"""
目標：
Deep Q learningの実装

参考文献 : 
[1] Pytorchで始める深層教科学習, p.171-179
[2] ゼロから作るDeep Learning4 強化学習編, p.233-252

ベンチマーク：
open gym, CartPole-v1

仕様：
基本的には[2]の例に則るが、オリジナルのフレームワークを使っているので、適宜[1]に倣ってPytorchに置き換える。
"""

# %%
#%%
# ライブラリのインポート

# 諸ライブラリ
import numpy as np
import random
import copy

# Open AI gymのライブラリ ← サポート終了してるらしい。
#import gym

# Pytorchライブラリ
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

# collectionsライブラリ
# collectionsについて　→　https://zenn.dev/giba/articles/38d7bf191cbc3f
from collections import deque

#%%
# 経験再生の実装, 参考[2]

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen = buffer_size) # Buffer の大きさを buffer_size に規定。
        self.batch_size = batch_size # Batch の大きさを batch_size に規定。

    def add(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done) # (状態, 行動, 報酬, 次状態, 終了フラッグ) を１つのデータとして結束。
        self.buffer.append(data) # 結束したデータを Buffer に格納。

    def __len__(self):
        return len(self.buffer) # buffer に格納されているデータの大きさを確認するコマンド。
    
    def get_batch(self):
        data = random.sample(self.buffer, self.batch_size) # Buffer から データを乱択して Mini Batch を生成。

        # Mini Batch からデータの各要素を整理してまとめ、各要素ごとに返す。
        state = np.stack([x[0] for x in data])
        action = np.array([x[1] for x in data])
        reward = np.array([x[2] for x in data])
        next_state = np.stack([x[3] for x in data])
        done = np.array([x[4] for x in data]).astype(np.int32)
        return state, action, reward, next_state, done


#%%
# ネットワークの実装, 参考[1]

class Net(nn.Module):
    def __init__(self, state_size, action_size):
        # 各層において、線形関数で順伝播させる。各ノードは変えても良い。
        super(Net, self).__init__()
        self.fc1 = nn.Linear(state_size, 64) # １層目 state_size → 64
        self.fc2 = nn.Linear(64, 64) # ２層目 64 → 64
        self.fc3 = nn.Linear(64, 64) # ３層目 64 → 64
        self.fc4 = nn.Linear(64, 64) # 4層目 64 → 64
        self.fc5 = nn.Linear(64, action_size) # 5層目 64 → action_size

    def forward(self, x):
        # 線形活性化関数で各層の入力を発火させる。
        x = F.softsign(self.fc1(x))
        x = F.softsign(self.fc2(x))
        x = F.softsign(self.fc3(x))
        x = F.softsign(self.fc4(x))
        x = self.fc5(x)
        return x
    

#%%
# エージェントの実装, 参考[2]

class DQNAgent:
    def __init__(self, state_size, action_size, policy):
        self.gamma = 0.98
        self.lr = 0.0005 # 学習率
        self.epsilon = 0.1
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.buffer_size = 10000
        self.batch_size = 32
        self.state_size = state_size
        self.action_size = action_size
        self.policy = policy

        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        self.qnet = Net(self.state_size, self.action_size) # Qnet
        self.qnet_target = Net(self.state_size, self.action_size) # ターゲットQnet
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=self.lr, betas=(0.9, 0.99), eps=1e-07) # 勾配を自動で降下させる。
        self.sync_qnet() # Qnet と ターゲットQnet を一致させるコマンド。

    # Qnet と ターゲットQnet を一致させるコマンド
    def sync_qnet(self):
        self.qnet_target.load_state_dict(self.qnet.state_dict())

    # 重みをロードする
    def load_state_dict(self, state_dict):
        self.qnet.load_state_dict(state_dict)

    # 行動を取得する。ただし、0 ~ action_size (> 0) の整数で取得される。ε - greedy法を用いる。
    def get_action(self, state):
        # epsilon - greedy方策
        if (self.policy == "epsilon_greedy"):
            if np.random.rand() < self.epsilon:
                return np.random.choice(self.action_size)
            else:
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                qs = self.qnet(state)
                act = torch.argmax(qs).item()
                # print(f"act={act}")
                return act
        
        # greedy方策
        elif (self.policy == "greedy"):
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            qs = self.qnet(state)
            act = torch.argmax(qs).item()
            # print(f"act={act}")
            return act
    
    # Qnetを更新する。
    def update(self):
        # Buffer が Mini Batch を構成できる大きさにならない限りは更新しない。
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Mini Batchを作成。
        state, action, reward, next_state, done = self.replay_buffer.get_batch()

        # 各データはNumpy配列で格納されているので、それをtensorに変換。
        state = np.array(state, dtype=np.float32)
        state = torch.tensor(state)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32)

        # Q値とターゲットQ値を個別に計算。
        qs = self.qnet(state)
        # print("qs={}".format(qs)) # デバック用
        q = qs[range(self.batch_size), action]

        next_qs = self.qnet_target(next_state)
        next_q = next_qs.max(dim=1)[0]
        target = reward + (1 - done) * self.gamma * next_q

        self.optimizer.zero_grad() # 勾配を初期化。
        loss = F.mse_loss(q, target) # 誤差を取得。
        loss.backward() # 誤差を逆伝播。
        self.optimizer.step() # 重みの更新。
    
    # 学習の進行度に合わせて epsilon を小さくしていく。
    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            


#%%
# CartPole環境でのDQNエージェントのトレーニング
"""
env = gym.make('CartPole-v1')
agent = DQNAgent()

num_episodes = 500
sync_freq = 10  # ターゲットネットワークの同期頻度
rewards = []

for episode in range(num_episodes):
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.get_action(state)
        next_state, reward, done, _, info = env.step(action)
        agent.replay_buffer.add(state, action, reward, next_state, done)
        agent.update()
        state = next_state
        total_reward += reward

    agent.update_epsilon()
    rewards.append(total_reward)

    if episode % sync_freq == 0:
        agent.sync_qnet()
    
    print(f"Episode {episode + 1}, Total Reward: {total_reward}")

env.close()


#%%
# 学習曲線のプロット

import matplotlib.pyplot as plt

plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Training Rewards')
plt.show()


# %%
# 学習後のエージェントのパフォーマンス評価

num_eval_episodes = 100
eval_rewards = []

for eval_episode in range(num_eval_episodes):
    state, _ = env.reset()  # 修正: state, _ として初期状態を取得
    total_reward = 0
    done = False

    while not done:
        action = agent.get_action(state)
        next_state, reward, done, _, info = env.step(action)  # 修正: infoを含めて5つの値を返す
        state = next_state
        total_reward += reward

    eval_rewards.append(total_reward)
    print(f"Evaluation Episode {eval_episode}, Total Reward: {total_reward}")

average_reward = np.mean(eval_rewards)
print(f"Average Reward over {num_eval_episodes} episodes: {average_reward}")
"""
# %%
