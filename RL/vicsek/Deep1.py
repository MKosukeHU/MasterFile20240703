# Deep Q-learningベースのIndependent-MARLを実装しよう

#%%
# Qネット

"""
 ▄▄▄▄▄▄▄▄▄▄   ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄       ▄▄▄▄▄▄▄▄▄▄▄ 
▐░░░░░░░░░░▌ ▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌     ▐░░░░░░░░░░░▌
▐░█▀▀▀▀▀▀▀█░▌▐░█▀▀▀▀▀▀▀▀▀ ▐░█▀▀▀▀▀▀▀▀▀ ▐░█▀▀▀▀▀▀▀█░▌     ▐░█▀▀▀▀▀▀▀█░▌
▐░▌       ▐░▌▐░▌          ▐░▌          ▐░▌       ▐░▌     ▐░▌       ▐░▌
▐░▌       ▐░▌▐░█▄▄▄▄▄▄▄▄▄ ▐░█▄▄▄▄▄▄▄▄▄ ▐░█▄▄▄▄▄▄▄█░▌     ▐░▌       ▐░▌
▐░▌       ▐░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌     ▐░▌       ▐░▌
▐░▌       ▐░▌▐░█▀▀▀▀▀▀▀▀▀ ▐░█▀▀▀▀▀▀▀▀▀ ▐░█▀▀▀▀▀▀▀▀▀      ▐░█▄▄▄▄▄▄▄█░▌
▐░▌       ▐░▌▐░▌          ▐░▌          ▐░▌               ▐░░░░░░░░░░░▌
▐░█▄▄▄▄▄▄▄█░▌▐░█▄▄▄▄▄▄▄▄▄ ▐░█▄▄▄▄▄▄▄▄▄ ▐░▌                ▀▀▀▀▀▀█░█▀▀ 
▐░░░░░░░░░░▌ ▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░▌                       ▐░▌  
 ▀▀▀▀▀▀▀▀▀▀   ▀▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀▀  ▀                         ▀   
"""

# 諸モジュール
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch import optim
from torch.nn import functional as F

# Qnetクラスの実装
class Qnet(nn.Module):
    def __init__(self, state_size, action_size):
        self.state = state_size
        self.action = action_size
        
        # 各層のノードの数を指定
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(self.state, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 96)
        self.fc4 = nn.Linear(96, 96)
        self.fc5 = nn.Linear(96, 64)
        self.fc6 = nn.Linear(64, 64)
        self.fc7 = nn.Linear(64, self.action)
    
    # 各層の活性化関数を指定
    def forward(self, x):
        x = Variable(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = self.fc7(x)
        return x
        
#%%
#経験再生の実装
#パッケージのインポート
from collections import deque #https://note.nkmk.me/python-collections-deque/
import random
import numpy as np

#Replay Buffer classの実装
class ReplayBuffer: #経験再生用のクラスを実装
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen = buffer_size) #bufferのサイズをbuffer_sizeに指定する
        self.batch_size = batch_size
        
    def add(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done) #bufferに追加するデータとして（状態、行動、報酬、次状態、終了判定）を入れる
        self.buffer.append(data) #データを追加
        
    def __len__(self):
        return len(self.buffer) #bufferに入っているデータ数を戻り値として返す？
    
    def get_batch(self): #mini-batch作成
        data = random.sample(self.buffer, self.batch_size) #bufferからself.bacth_size分のデータを乱択してmini-batchに保存する
        
        state = np.stack([x[0] for x in data]) #作成したmini-bacthからデータの種類ごとに仕分けする
        action = np.array([x[1] for x in data])
        reward = np.array([x[2] for x in data])
        next_state = np.stack([x[3] for x in data])
        done = np.array([x[4] for x in data]).astype(np.int32)
        return state, action, reward, next_state, done


#%%
#エージェントの実装
import copy
import torch

#ndarrayからTensorへの変換方法：tensor = torch.from_numpy(array)
#今回はデータをndarrayの配列で取得しているので、一旦Tensorに変換してからNNの計算をする必要がある。

#dqn_agentの実装
class dqn_agent:
    def __init__(self, action_size, state_size, policy, num_agent):
        #各定数を定義
        self.gamma = 0.98 #報酬の割引率
        self.lr = 0.0005 #学習率
        self.epsilon = 0.1 #epsilon-greedy法で用いる
        self.buffer_size = 10000 #bufferのサイズ
        self.batch_size = 32 #mini_batchのサイズは32に制限
        self.action_size = action_size #出力層の次元はaction_sizeを用いる
        self.D_in = state_size #入力層は状態の次元
        self.policy = policy
        self.num_agent = num_agent
        
        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        self.qnet = Qnet(self.D_in, self.action_size) #本命Qネット
        self.qnet_target = Qnet(self.D_in, self.action_size) #教師信号生成用のターゲットQネット
        self.optimizer = optim.Adam(self.qnet.parameters(), lr = self.lr, betas = (0.9, 0.99), eps = 1e-07) #betasはこれが常套らしい、epsはゼロ除算回避用
        self.optimizer.step() #qnetのシナプス結合荷重（重み）の更新 

    # 重みをロードする
    def load_state_dict(self, state_dict):
        self.qnet.load_state_dict(state_dict)
        
    #target_netとの重みの同期    
    def sync_qnet(self):
        self.qnet_target = copy.deepcopy(self.qnet)
    
    #Q値を取得
    def take_Qvalue(self, state):
        state = np.array(state)  # state を np.ndarray に変換
        state = torch.from_numpy(state).float()  # NumPy配列をTensorに変換
        state = state.unsqueeze(0)  # バッチ次元を追加
        qs = self.qnet(state)
        return qs
        
    #行動取得
    def get_action(self, state):
        if self.policy == 'epsilon_greedy':
            state = np.array(state)  # state を np.ndarray に変換
            state = torch.from_numpy(state).float()  # NumPy配列をTensorに変換
            if np.random.rand() < self.epsilon:
                return np.random.choice(self.action_size) #epsilonの確率でランダムな行動選択
            else:
                #state = state.unsqueeze(0)  # バッチ次元を追加
                qs = self.qnet(state)
                act = torch.argmax(qs)
                return act.tolist()  #それ以外ではあるstateに対して行動価値関数が最大となる行動選択を行う
            
        elif self.policy == 'greedy':
            state = np.array(state)  # state を np.ndarray に変換
            state = torch.from_numpy(state).float()  # NumPy配列をTensorに変換
            #state = state.unsqueeze(0)  # バッチ次元を追加
            qs = self.qnet(state)
            act = torch.argmax(qs)
            return act.tolist()  #それ以外ではあるstateに対して行動価値関数が最大となる行動選択を行う
        
    #qnetの更新
    def update(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer) < self.batch_size: #bufferに入ったデータ数がmini-batchの規定サイズを超えない限りはupdateはしない
            return
        
        state, action, reward, next_state, done = self.replay_buffer.get_batch() #mini-batchからデータを取り出す
        
        #データをシミュレーション環境から取得するときにデータ型はnumpy配列になっているのでそれをTensorに変換する必要がある
        state = torch.from_numpy(state).float() # NumPy配列をTensorに変換
        reward = torch.from_numpy(reward).float()  # NumPy配列をTensorに変換
        next_state = torch.from_numpy(next_state).float() # NumPy配列をTensorに変換
        done = torch.from_numpy(done.astype(np.float32)).float()  # NumPy配列をTensorに変換
        
        qs = self.qnet(state)
        q = qs[np.arange(self.batch_size), action] #行動価値関数を算出
        
        next_qs = self.qnet_target(next_state)
        next_q = next_qs.max(axis=1).values  # 最大値を取得
        
        if next_q.size() != reward.size():
            print('reward={}'.format(reward.size()))
            print('next_q={}'.format(next_q.size()))
        
        target = reward + (1 - done) + self.gamma * next_q #教師信号を算出
        
        criterion = nn.MSELoss()
        loss = criterion(q, target) #最小二乗誤差を計算
            
        self.optimizer.zero_grad() #勾配の初期化
        loss.backward() #誤差を逆伝播
        self.optimizer.step() #重みの更新


#%%
# 環境
"""
 ▄▄▄▄▄▄▄▄▄▄▄  ▄▄        ▄  ▄               ▄ 
▐░░░░░░░░░░░▌▐░░▌      ▐░▌▐░▌             ▐░▌
▐░█▀▀▀▀▀▀▀▀▀ ▐░▌░▌     ▐░▌ ▐░▌           ▐░▌ 
▐░▌          ▐░▌▐░▌    ▐░▌  ▐░▌         ▐░▌  
▐░█▄▄▄▄▄▄▄▄▄ ▐░▌ ▐░▌   ▐░▌   ▐░▌       ▐░▌   
▐░░░░░░░░░░░▌▐░▌  ▐░▌  ▐░▌    ▐░▌     ▐░▌    
▐░█▀▀▀▀▀▀▀▀▀ ▐░▌   ▐░▌ ▐░▌     ▐░▌   ▐░▌     
▐░▌          ▐░▌    ▐░▌▐░▌      ▐░▌ ▐░▌      
▐░█▄▄▄▄▄▄▄▄▄ ▐░▌     ▐░▐░▌       ▐░▐░▌       
▐░░░░░░░░░░░▌▐░▌      ▐░░▌        ▐░▌        
 ▀▀▀▀▀▀▀▀▀▀▀  ▀        ▀▀          ▀         
"""

import numpy as np

#シミュレーション環境の実装   
import numpy as np
import random

class Enviroment:
    def __init__(self, action_size, num_agent, perceptual_range, lx_out, ly_out, v0, eta):
        self.action_size = action_size
        self.num_agent = num_agent
        self.radius = perceptual_range
        self.lx_out = lx_out
        self.ly_out = ly_out
        self.v0 = v0
        self.eta = eta
    
    def generator(self):
        return np.random.uniform(-1.0, 1.0)
        
    def pBC(self, x_, l_):
        return np.where(x_ < 0, x_ + l_, np.where(x_ > l_, x_ - l_, np.where((-l_ <= x_) & (x_ <= l_), x_, 0)))
    
    def pos_to_vec(self, pos_):
        pos_trans = pos_.T
        x_vec = pos_trans[0]
        y_vec = pos_trans[1]
        angle_vec = pos_trans[2]
        R_vec = pos_trans[3]
        return x_vec, y_vec, angle_vec, R_vec
    
    def num_to_theta(self, a):
        action_list = np.linspace(start=-(3/16)*np.pi, stop=(3/16)*np.pi, num=self.action_size)
        theta_add = action_list[a] # a in {0,1,2,3,4,5,6}
        return theta_add
    
    def get_s_next(self, s, a):
        x, y, theta, R = self.pos_to_vec(s) # 各成分に分解
        theta_add =  self.num_to_theta(a)
        
        # 次時刻の状態を計算
        theta_next = theta + theta_add + np.random.uniform(low=-(1/2), high=(1/2), size=self.num_agent) * self.eta
        theta_next = np.arctan2(np.sin(theta_next), np.cos(theta_next)) #　正規化
        x_next = x + self.v0 * np.cos(theta_next)
        y_next = y + self.v0 * np.sin(theta_next)
        
        # 周期境界条件を適用
        x_next = self.pBC(x_next, self.lx_out)
        y_next = self.pBC(y_next, self.ly_out)
        
        return np.array([x_next, y_next, theta_next, R]).T
    
    def cul_neighbor(self, s):
        x_, y_, _, R = self.pos_to_vec(s)
        R_m = np.tile(R, (self.num_agent, 1)).T
        dx_ = np.abs(x_[np.newaxis, :] - x_[:, np.newaxis])
        dx_ = np.where(dx_ > self.lx_out / 2, dx_ - self.lx_out, dx_)
        dy_ = np.abs(y_[np.newaxis, :] - y_[:, np.newaxis])
        dy_ = np.where(dy_ > self.ly_out / 2, dy_ - self.ly_out, dy_)
        dis_cul = np.sqrt(dx_**2 + dy_**2) < R_m
        output = dis_cul # - np.eye(self.num_agent), No6
        return output
    
    def cul_dxdy(self, s):
        x_, y_, _, R = self.pos_to_vec(s)
        dx_ = np.abs(x_[np.newaxis, :] - x_[:, np.newaxis])
        dx_ = np.where(dx_ > self.lx_out / 2, dx_ - self.lx_out, dx_)
        dy_ = np.abs(y_[np.newaxis, :] - y_[:, np.newaxis])
        dy_ = np.where(dy_ > self.ly_out / 2, dy_ - self.ly_out, dy_)
        return dx_, dy_
    
    def generate_position(self):
        x = np.random.uniform(low=0, high=self.lx_out, size=self.num_agent)
        y = np.random.uniform(low=0, high=self.ly_out, size=self.num_agent)
        theta = np.random.uniform(low=-np.pi, high=np.pi, size=self.num_agent)
        R = np.full(self.num_agent, self.radius)
        return np.array([x, y, theta, R]).T
    
    def reset(self):
        reset_position = self.generate_position()
        return reset_position
    
    def step(self, s, a):
        past_position = np.array(s)
        ad_m_past = self.cul_neighbor(past_position)
        x, y, theta, R = self.pos_to_vec(past_position)
        theta_past = theta * ad_m_past
        #dist_past = self.dist(past_position)
        #neighbor_past = self.cul_neighbor(past_position)
        #num_neighbor_past = np.sum(neighbor_past, axis=1)
        next_position = self.get_s_next(s, a)
        #ad_m_next = self.cul_neighbor(next_position)
        #x_, y_, theta_, R_ = self.pos_to_vec(next_position)
        #theta_next = theta_ * ad_m_next
        #dist_next = self.dist(next_position)
        #neighbor_next = self.cul_neighbor(next_position)
        #num_neighbor_next = np.sum(neighbor_next, axis=1)
    
        #neighbor_matrix_past = self.cul_neighbor(past_position).sum(axis=1)
        #neighbor_matrix_next = self.cul_neighbor(next_position).sum(axis=1)    
        #reward = np.where(neighbor_matrix_past > neighbor_matrix_next, 1, 0)

        order_past = (1 / self.num_agent) * np.sqrt((np.sum(np.cos(theta_past), axis=1))**2 + np.sum(np.sin(theta_past), axis=1)**2)
        #order_next = (1 / self.num_agent) * np.sqrt((np.sum(np.cos(theta_next), axis=1))**2 + np.sum(np.sin(theta_next), axis=1)**2)
        reward = order_past

        #reward = np.where(dist_past > dist_next, 1, 0)
        #index = np.where(num_neighbor_next >= num_neighbor_past, 1, 0)
        #reward = num_neighbor_next * index
        #reward = np.where(num_neighbor_past > 0, 1, 0)
        #reward = np.where(dist_past < 2.5, 1, 0)
        return np.array(next_position), reward, False
    
    # 先行研究におけるstate関数
    def arg(self, s):
        x, y, theta, _ = self.pos_to_vec(s)
        dis_cul = self.cul_neighbor(s)
        #dis_cul = self.see_foward(s) # sim5 is see_foward
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
        rota_v = np.tile(rota, (self.num_agent, 1, 1))
        vTop = v.reshape((self.num_agent, 2, 1))
        rota_dot_vTop = np.matmul(rota_v, vTop).reshape((self.num_agent, 2))
        P_dot_vTop = (P * rota_dot_vTop).sum(axis=1)
        
        # 返り値を計算
        naiseki = (P_normalize * v).sum(axis=1)
        arccos = np.arccos(naiseki)
        output = np.where(P_dot_vTop > 0, arccos, -arccos) # 場合分け
        return output
    
    def dist(self, positions):
        neighbor_matrix = self.cul_neighbor(positions)
        num_neighbor = np.sum(neighbor_matrix, axis=1)
        dx, dy = self.cul_dxdy(positions)
        dx_neighbor = np.where(neighbor_matrix == 1, dx, 0)
        dy_neighbor = np.where(neighbor_matrix == 1, dy, 0)
        dx_mean = np.sum(dx_neighbor, axis=1) / num_neighbor
        dy_mean = np.sum(dy_neighbor, axis=1) / num_neighbor
        dist_mean = np.sqrt(dx_mean**2 + dy_mean**2)
        return dist_mean
    
    def generate_state(self, position): # position = num_agent x 3
        #output_state_arg = self.arg(position)

        output_state_dist = self.dist(position)

        nei_index = self.cul_neighbor(position)
        num_neighbor = np.sum(nei_index, axis=1)
        dx_, dy_ = self.cul_dxdy(position)
        output_state_dx = np.where(num_neighbor != 0, np.sum(dx_, axis=1) / num_neighbor, 0)
        output_state_dy = np.where(num_neighbor != 0, np.sum(dy_, axis=1) / num_neighbor, 0)
        v_relative = self.v0[np.newaxis,:] - self.v0[:,np.newaxis]
        v_relative_neighbor = np.where(nei_index == True, v_relative, 0)
        output_state_v = np.sum(v_relative_neighbor, axis=1) / num_neighbor
        
        output_state = np.array([output_state_dx, output_state_dx, output_state_dist, output_state_v]).T
        return output_state # len(output_state) = num_agent x 2
    

#%%
# 学習
"""
 ▄            ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄  ▄▄        ▄ 
▐░▌          ▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░▌      ▐░▌
▐░▌          ▐░█▀▀▀▀▀▀▀▀▀ ▐░█▀▀▀▀▀▀▀█░▌▐░█▀▀▀▀▀▀▀█░▌▐░▌░▌     ▐░▌
▐░▌          ▐░▌          ▐░▌       ▐░▌▐░▌       ▐░▌▐░▌▐░▌    ▐░▌
▐░▌          ▐░█▄▄▄▄▄▄▄▄▄ ▐░█▄▄▄▄▄▄▄█░▌▐░█▄▄▄▄▄▄▄█░▌▐░▌ ▐░▌   ▐░▌
▐░▌          ▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░▌  ▐░▌  ▐░▌
▐░▌          ▐░█▀▀▀▀▀▀▀▀▀ ▐░█▀▀▀▀▀▀▀█░▌▐░█▀▀▀▀█░█▀▀ ▐░▌   ▐░▌ ▐░▌
▐░▌          ▐░▌          ▐░▌       ▐░▌▐░▌     ▐░▌  ▐░▌    ▐░▌▐░▌
▐░█▄▄▄▄▄▄▄▄▄ ▐░█▄▄▄▄▄▄▄▄▄ ▐░▌       ▐░▌▐░▌      ▐░▌ ▐░▌     ▐░▐░▌
▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░▌       ▐░▌▐░▌       ▐░▌▐░▌      ▐░░▌
 ▀▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀▀  ▀         ▀  ▀         ▀  ▀        ▀▀  
"""    

import numpy as np
import os
import torch
import time

class Learn:
    def __init__(self, action_size, state_size, policy, num, R, v, lx, ly, eta):
        self.action_size = action_size # 行動数
        self.state_size = state_size # 状態数
        self.policy = policy # 方策
        self.num = num # エージェント数
        self.R = R # 視野
        self.v = v # 自己駆動速度
        self.lx = lx # 空間サイズ
        self.ly = ly
        self.eta = eta # ノイズ強度
        
        self.agents = [dqn_agent(self.action_size, self.state_size, self.policy, self.num) for i in range(self.num)]
        self.env = Enviroment(self.action_size, self.num, self.R, self.lx, self.ly, self.v, self.eta)

    def save(self, tag):
        save_dir = 'my_model/'

        # ディレクトリが存在しない場合は作成する
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 1. モデルの保存ディレクトリを指定する
        save_dir = 'my_model/'

        # 2. モデルの重みを取得する
        model = [self.agents[i].qnet.state_dict() for i in range(self.num)]
            
        # 4. ファイル名の指定
        file_name = 'my_model' + str(tag) + '.pth'

        # 3. 重みを保存する
        torch.save(model, save_dir + file_name)
        print('File name is {}'.format(file_name))
        
    def run(self, tag, episodes, Tmax):
        reward_episodes = []
        size_episodes = []
        for ep in range(episodes):
            tortal_reward = 0
            positions = self.env.reset()
            
            for t in range(Tmax):
                state = self.env.generate_state(positions)
                action = np.array([self.agents[i].get_action(state[i]) for i in range(self.num)])
                #print('episode={}, t={}, action={}'.format(ep, t, action))
                
                positions_next, reward, _ = self.env.step(positions, action)
                state_next = self.env.generate_state(positions_next)

                positions = positions_next

                for i in range(self.num):
                    self.agents[i].update(state[i], action[i], reward[i], state_next[i], done=False)
                
                tortal_reward += float(np.mean(reward))
                
                if t % (Tmax / 100) == 0:
                    print("progress={}%".format(int(t/Tmax * 100)), end="\r")
                    time.sleep(0.1)
                
            if ep % 20 == 0:
                for i in range(self.num):
                    self.agents[i].sync_qnet()
            
            print('Ep={}, Reward={:.3f}'.format(ep+1, tortal_reward))
            
            reward_episodes.append(tortal_reward)
            
        self.save(tag)
        
        return reward_episodes
    
    def simulate(self, model, Tmax):
        agents = [dqn_agent(self.action_size, self.state_size, 'greedy', self.num) for i in range(num)]
        env = Enviroment(self.action_size, self.num, self.R, self.lx, self.ly, self.v, self.eta)

        for i in range(self.num):
            agents[i].load_state_dict(model[i])

        positions = env.generate_position()
        pos_record = [positions]
        reward_record = []
        for t in range(Tmax):
            state = self.env.generate_state(positions)
            action = np.array([self.agents[i].get_action(state[i]) for i in range(self.num)])
            #print('episode={}, t={}, action={}'.format(ep, t, action))
                
            positions_next, reward, _ = self.env.step(positions, action)

            positions = positions_next
            pos_record.append(positions)

            reward_record.append(np.mean(reward))
                
            if t % (Tmax / 100) == 0:
                print("progress={}%".format(int(t/Tmax * 100)), end="\r")
                time.sleep(0.1)
        return pos_record, reward_record


#%%
# 学習の実行
"""
 ▄▄▄▄▄▄▄▄▄▄▄  ▄         ▄  ▄▄        ▄ 
▐░░░░░░░░░░░▌▐░▌       ▐░▌▐░░▌      ▐░▌
▐░█▀▀▀▀▀▀▀█░▌▐░▌       ▐░▌▐░▌░▌     ▐░▌
▐░▌       ▐░▌▐░▌       ▐░▌▐░▌▐░▌    ▐░▌
▐░█▄▄▄▄▄▄▄█░▌▐░▌       ▐░▌▐░▌ ▐░▌   ▐░▌
▐░░░░░░░░░░░▌▐░▌       ▐░▌▐░▌  ▐░▌  ▐░▌
▐░█▀▀▀▀█░█▀▀ ▐░▌       ▐░▌▐░▌   ▐░▌ ▐░▌
▐░▌     ▐░▌  ▐░▌       ▐░▌▐░▌    ▐░▌▐░▌
▐░▌      ▐░▌ ▐░█▄▄▄▄▄▄▄█░▌▐░▌     ▐░▐░▌
▐░▌       ▐░▌▐░░░░░░░░░░░▌▐░▌      ▐░░▌
 ▀         ▀  ▀▀▀▀▀▀▀▀▀▀▀  ▀        ▀▀ 
"""

import numpy as np
action_size = 7 # 行動数
state_size = 4 # 状態数
policy = 'epsilon_greedy' # 方策
num = 50 # エージェント数
R = 1 # 視野
v = np.ones(num) * 0.5 # 自己駆動速度
l = int(np.sqrt(num / 2)) # 密度依存の空間サイズ
lx = l # 空間サイズ
ly = l
eta = 0 # ノイズ強度

episodes = 1000
Tmax = 500
tag = 'No9'

machine = Learn(action_size, state_size, policy, num, R, v, lx, ly, eta)
reward_record = machine.run(tag, episodes, Tmax)

#%%
# データの保存
import numpy as np
data_tag = 'data/' + str(tag)
np.savetxt(data_tag, reward_record)

#%%
# 報酬合計の推移
import numpy as np
import matplotlib.pyplot as plt

data_reward = np.loadtxt(data_tag)

plt.plot(np.arange(len(data_reward)), data_reward, 'o-', color='red')
plt.xlabel('Episode')
plt.ylabel('Total reward')
plt.show()


#%%
# エージェントによるシミュレーション

"""
 ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄  ▄▄       ▄▄  ▄         ▄  ▄            ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄ 
▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░▌     ▐░░▌▐░▌       ▐░▌▐░▌          ▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌
▐░█▀▀▀▀▀▀▀▀▀  ▀▀▀▀█░█▀▀▀▀ ▐░▌░▌   ▐░▐░▌▐░▌       ▐░▌▐░▌          ▐░█▀▀▀▀▀▀▀█░▌ ▀▀▀▀█░█▀▀▀▀ ▐░█▀▀▀▀▀▀▀▀▀ 
▐░▌               ▐░▌     ▐░▌▐░▌ ▐░▌▐░▌▐░▌       ▐░▌▐░▌          ▐░▌       ▐░▌     ▐░▌     ▐░▌          
▐░█▄▄▄▄▄▄▄▄▄      ▐░▌     ▐░▌ ▐░▐░▌ ▐░▌▐░▌       ▐░▌▐░▌          ▐░█▄▄▄▄▄▄▄█░▌     ▐░▌     ▐░█▄▄▄▄▄▄▄▄▄ 
▐░░░░░░░░░░░▌     ▐░▌     ▐░▌  ▐░▌  ▐░▌▐░▌       ▐░▌▐░▌          ▐░░░░░░░░░░░▌     ▐░▌     ▐░░░░░░░░░░░▌
 ▀▀▀▀▀▀▀▀▀█░▌     ▐░▌     ▐░▌   ▀   ▐░▌▐░▌       ▐░▌▐░▌          ▐░█▀▀▀▀▀▀▀█░▌     ▐░▌     ▐░█▀▀▀▀▀▀▀▀▀ 
          ▐░▌     ▐░▌     ▐░▌       ▐░▌▐░▌       ▐░▌▐░▌          ▐░▌       ▐░▌     ▐░▌     ▐░▌          
 ▄▄▄▄▄▄▄▄▄█░▌ ▄▄▄▄█░█▄▄▄▄ ▐░▌       ▐░▌▐░█▄▄▄▄▄▄▄█░▌▐░█▄▄▄▄▄▄▄▄▄ ▐░▌       ▐░▌     ▐░▌     ▐░█▄▄▄▄▄▄▄▄▄ 
▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░▌       ▐░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░▌       ▐░▌     ▐░▌     ▐░░░░░░░░░░░▌
 ▀▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀▀  ▀         ▀  ▀▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀▀  ▀         ▀       ▀       ▀▀▀▀▀▀▀▀▀▀▀ 
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

Tmax_sim = 400

model = torch.load('my_model/my_modelNo9.pth')
pos_reco, rew_reco = machine.simulate(model, Tmax_sim)


# %%
# アニメーションの作成
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML

# キャンバスを設置
fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot()

ax1.set_xlim(0, lx)
ax1.set_ylim(0, ly)

#アップデート関数
def update1(i):
    ax1.clear()
    ax1.set_xlim(0, lx)
    ax1.set_ylim(0, ly)

    data = pos_reco[i].T
    data_x, data_y, data_theta = data[0], data[1], data[2]

    ax1.quiver(data_x, data_y, np.cos(data_theta), np.sin(data_theta), color='black')
    ax1.set_title('use model is No8, time={}'.format(i+1), fontsize=14)
    # 進捗状況を出力する
    progress_rate = (i+1)/Tmax*100
    print("Animation progress={:.3f}%".format(progress_rate), end='\r')
        
#アニメーション作成とgif保存
ani = animation.FuncAnimation(fig, update1, frames=range(Tmax_sim))
    
#グラフ表示
plt.show()

#アニメーションを表示
HTML(ani.to_jshtml())

# アニメーションの保存
ani.save('animation/model=No8.mp4', writer='ffmpeg')


# %%
import numpy as np
a = np.arange(3)
b = np.eye(3)
c = a*b
print(a, b, c)
# %%
