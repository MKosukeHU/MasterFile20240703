"""

"""


#%%
# テーブル型のQ学習を実装
import numpy as np
import random

class QTableAgent:
    def __init__(self, state_size, action_size, num_agent, policy, max_episodes, gamma=0.98, learning_rate=0.005, epsilon=0.002):
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = np.zeros((num_agent, state_size, action_size))
        self.policy = policy
        self.index = np.arange(num_agent)
        self.num_agent = num_agent
        self.gamma = gamma #報酬の割引率
        self.learning_rate = learning_rate #学習率
        self.epsilon = epsilon #epsilon-greedy法で用いる
        self.max_episodes = max_episodes
        self.bins = np.linspace(start=-np.pi, stop=np.pi, num=state_size) # output_stateの値域に注意して設定せよ

    def update(self, state, action, reward, next_state, done):
        # Q学習の更新式
        state_index = np.digitize(state, self.bins) - 1
        next_state_index = np.digitize(next_state, self.bins) - 1
        q_sa = self.q_table[self.index, state_index, action]
        q_max = np.max(self.q_table[self.index, next_state_index], axis=1)
        self.q_table[self.index, state_index, action] = (1 - self.learning_rate) * q_sa + self.learning_rate * (reward + self.gamma * q_max)
        
    def get_action(self, state, episode):
        state_index = np.digitize(state, self.bins) - 1
        if self.policy == 'greedy':
            qs = self.q_table[self.index, state_index]
            action = np.argmax(qs, axis=1)
        elif self.policy == 'epsilon_greedy':
            if np.random.rand() < self.epsilon * (self.max_episodes - episode) / self.max_episodes:
                action = np.random.choice(self.action_size, size=self.num_agent)
            else:
                qs = self.q_table[self.index, state_index]
                action = np.argmax(qs, axis=1)
        return action
    
    def load_q_table(self, file_path):
        # ファイルからQテーブルを読み込む
        loaded_q_table = np.load(file_path)
        # エージェントのQテーブルに読み込んだQテーブルを設定
        self.q_table = loaded_q_table
    
    def get_q_table(self):
        return self.q_table


#%%
# テーブル型のQ学習を実装
import numpy as np
import random

class QTableAgent_min:
    def __init__(self, state_size, action_size, num_agent, policy, max_episodes, gamma=0.98, learning_rate=0.005, epsilon=0.002):
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = np.zeros((num_agent, state_size, action_size))
        self.policy = policy
        self.index = np.arange(num_agent)
        self.num_agent = num_agent
        self.gamma = gamma #報酬の割引率
        self.learning_rate = learning_rate #学習率
        self.epsilon = epsilon #epsilon-greedy法で用いる
        self.max_episodes = max_episodes
        self.bins = np.linspace(start=-np.pi, stop=np.pi, num=state_size) # output_stateの値域に注意して設定せよ

    def update(self, state, action, reward, next_state, done):
        # Q学習の更新式
        state_index = np.digitize(state, self.bins) - 1
        next_state_index = np.digitize(next_state, self.bins) - 1
        q_sa = self.q_table[self.index, state_index, action]
        self.q_table[self.index, state_index, action] += self.learning_rate * (reward - q_sa)
        
    def get_action(self, state, episode):
        state_index = np.digitize(state, self.bins) - 1
        if self.policy == 'greedy':
            qs = self.q_table[self.index, state_index]
            action = np.argmin(qs, axis=1)
        elif self.policy == 'epsilon_greedy':
            if np.random.rand() < self.epsilon * (self.max_episodes - episode) / self.max_episodes:
                action = np.random.choice(self.action_size, size=self.num_agent)
            else:
                qs = self.q_table[self.index, state_index]
                action = np.argmin(qs, axis=1)
        return action
    
    def load_q_table(self, file_path):
        # ファイルからQテーブルを読み込む
        loaded_q_table = np.load(file_path)
        # エージェントのQテーブルに読み込んだQテーブルを設定
        self.q_table = loaded_q_table
    
    def get_q_table(self):
        return self.q_table
