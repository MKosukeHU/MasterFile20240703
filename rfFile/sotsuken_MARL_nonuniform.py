#%%
# テーブル型のQ学習を実装
import numpy as np
import random

class QTableAgent:
    def __init__(self, state_size, action_size, num_agent, policy):
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = np.zeros((num_agent, state_size, action_size))
        self.policy = policy
        self.index = np.arange(num_agent)
        self.num_agent = num_agent
        self.gamma = 0.98 #報酬の割引率
        self.learning_rate = 0.005 #学習率
        self.epsilon = 0.002 #epsilon-greedy法で用いる
        self.bins = np.linspace(start=-np.pi, stop=np.pi, num=state_size) # output_stateの値域に注意して設定せよ

    def update(self, state, action, reward, next_state, done):
        # Q学習の更新式
        state_index = np.digitize(state, self.bins) - 1
        #next_state_index = np.digitize(next_state, self.bins) - 1
        #target = reward + (1 - done) * self.gamma * np.max(self.q_table[self.index, next_state_index], axis=1)
        self.q_table[self.index, state_index, action] += self.learning_rate * (reward - self.q_table[self.index, state_index, action])
        
    def get_action(self, state, episode):
        state_index = np.digitize(state, self.bins) - 1
        if self.policy == 'greedy':
            qs = self.q_table[self.index, state_index]
            action = np.argmin(qs, axis=1)
        elif self.policy == 'epsilon_greedy':
            if np.random.rand() < 1 - self.epsilon * (episode - 1) if episode < 900 else 0.:
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
    
    def get_QTable(self):
        return self.q_table
    

#%%
class Enviroment_vicsek_uniform:
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
        return x_vec, y_vec, angle_vec
    
    def num_to_theta(self, a):
        action_list = np.linspace(start=-(3/16)*np.pi, stop=(3/16)*np.pi, num=self.action_size)
        theta_add = action_list[a] # a in {0,1,2,3,4,5,6}
        return theta_add
    
    def get_s_next(self, s, a):
        x, y, theta = self.pos_to_vec(s) # 各成分に分解
        theta_add =  self.num_to_theta(a)
        
        # 次時刻の状態を計算
        theta_next = theta + theta_add + np.random.uniform(low=-(1/2), high=(1/2), size=self.num_agent) * self.eta
        theta_next = np.arctan2(np.sin(theta_next), np.cos(theta_next)) #　正規化
        x_next = x + self.v0 * np.cos(theta_next)
        y_next = y + self.v0 * np.sin(theta_next)
        
        # 周期境界条件を適用
        x_next = self.pBC(x_next, self.lx_out)
        y_next = self.pBC(y_next, self.ly_out)
        
        return np.array([x_next, y_next, theta_next]).T
    
    def cul_neighbor(self, s):
        x_, y_, _ = self.pos_to_vec(s)
        dx_ = np.abs(x_[np.newaxis, :] - x_[:, np.newaxis])
        dx_ = np.where(dx_ > self.lx_out / 2, dx_ - self.lx_out, dx_)
        dy_ = np.abs(y_[np.newaxis, :] - y_[:, np.newaxis])
        dy_ = np.where(dy_ > self.ly_out / 2, dy_ - self.ly_out, dy_)
        dis_cul = np.sqrt(dx_**2 + dy_**2) < self.radius
        return dis_cul
    
    def take_order_and_dencity(self, s):
        # 隣接行列を計算
        dis_cul = self.cul_neighbor(s)
        _, _, theta = self.pos_to_vec(s)
        # 各エージェントのcosとsin
        cos = np.cos(theta)
        sin = np.sin(theta)

        # cosとsinを距離行列にマスクして取得
        cos_neighbor = np.dot(dis_cul, cos)
        sin_neighbor = np.dot(dis_cul, sin)

        # 各エージェントの隣接エージェント数
        num_neighbors = dis_cul.sum(axis=1)

        # ソートして正規化したlocal_orderを計算
        order_norm = np.sqrt(cos_neighbor**2 + sin_neighbor**2)
        local_order = np.where(num_neighbors != 0, order_norm / num_neighbors, 0)
        
        # 密度を計算
        dencity = num_neighbors / (self.radius**2 * np.pi)

        # grobal orderを計算
        grobal_order = (1 / self.num_agent) * np.sqrt(np.sum(np.cos(theta))**2 + np.sum(np.sin(theta))**2)
            
        return grobal_order, local_order, dencity
    
    def generate_position(self):
        x = np.random.uniform(low=0, high=self.lx_out, size=self.num_agent)
        y = np.random.uniform(low=0, high=self.ly_out, size=self.num_agent)
        theta = np.random.uniform(low=-np.pi, high=np.pi, size=self.num_agent)
        return np.array([x, y, theta]).T
    
    def reset(self):
        reset_position = self.generate_position()
        return reset_position
    
    def step(self, s, a):
        past_position = np.array(s)
        next_position = self.get_s_next(s, a)
    
        neighbor_matrix_past = self.cul_neighbor(past_position).sum(axis=1)
        neighbor_matrix_next = self.cul_neighbor(next_position).sum(axis=1)    
        reward = np.where(neighbor_matrix_past > neighbor_matrix_next, 1, 0)
        
        #_, reward, _ = self.take_order_and_dencity(s)
            
        return np.array(next_position), reward, False
    
    # 先行研究におけるstate関数
    def arg(self, x, y, theta):
        dx = x[np.newaxis,:] - x[:,np.newaxis] # num_agent x num_agent
        dy = y[np.newaxis,:] - y[:,np.newaxis]
        dis_cul = dx**2 + dy**2 < self.R**2 
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
        rota_v = np.tile(rota, (self.N, 1, 1))
        vTop = v.reshape((self.N, 2, 1))
        rota_dot_vTop = np.matmul(rota_v, vTop).reshape((self.N, 2))
        P_dot_vTop = (P * rota_dot_vTop).sum(axis=1)
        
        # 返り値を計算
        naiseki = (P_normalize * v).sum(axis=1)
        arccos = np.arccos(naiseki)
        output = np.where(P_dot_vTop > 0, arccos, -arccos) # 場合分け
        return output
    
    def generate_state(self, position): # position = num_agent x 3
        output_state = self.arg(position)
        
        return output_state # len(output_state) = num_agent


#%%
#シミュレーション環境の実装   
import numpy as np
import random

class Enviroment_vicsek:
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
    
    def see_foward(self, position):
        x, y, theta, R = self.pos_to_vec(position)
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dist = np.sqrt(dx**2 + dy**2)
        index_neighbor = self.cul_neighbor(position)
        
        dot = (dx * np.cos(theta).reshape((self.num_agent,1)) + dy * np.sin(theta).reshape((self.num_agent,1))) / dist
        index_insight = np.arccos(dot) < (3/16)*np.pi
        
        index_insight_neighbor = index_neighbor * index_insight
        return index_insight_neighbor
    
    """
    def arg2(self, position):
        x, y, theta, R = self.pos_to_vec(position)
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dist = np.sqrt(dx**2 + dy**2)
        dot = (dx * np.cos(theta).reshape((self.num_agent,1)) + dy * np.sin(theta).reshape((self.num_agent,1))) / dist
        
        dx_rot = np.cos(np.pi/2) * dx + (-np.sin(np.pi/2)) * dy
        dy_rot = np.sin(np.pi/2) * dx + np.cos(np.pi/2) * dy
        dist_rot = np.sqrt(dx_rot**2 + dy_rot**2)
        dot_rot = (dx_rot * np.cos(theta).reshape((self.num_agent,1)) + dy_rot * np.sin(theta).reshape((self.num_agent,1))) / dist_rot
        
        output_pre = np.where(dot_rot > 0, np.arccos(dot), -np.arccos(dot))
        
        index = self.see_foward(position)
        output_index = np.where(index, output_pre, np.nan)
        output = np.nanmean(output_pre)
        return output
    """
    
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
        return dis_cul
    
    def take_order_and_dencity(self, s):
        # 隣接行列を計算
        dis_cul = self.cul_neighbor(s) # sim1~3, 5
        #dis_cul = self.see_foward(s) # sim4
        _, _, theta, R = self.pos_to_vec(s)
        # 各エージェントのcosとsin
        cos = np.cos(theta)
        sin = np.sin(theta)

        # cosとsinを距離行列にマスクして取得
        cos_neighbor = np.dot(dis_cul, cos)
        sin_neighbor = np.dot(dis_cul, sin)

        # 各エージェントの隣接エージェント数
        num_neighbors = dis_cul.sum(axis=1)

        # ソートして正規化したlocal_orderを計算
        order_norm = np.sqrt(cos_neighbor**2 + sin_neighbor**2)
        local_order = np.where(num_neighbors != 0, order_norm / num_neighbors, 0)
        
        # 密度を計算
        dencity = num_neighbors / (R**2 * np.pi)

        # grobal orderを計算
        grobal_order = (1 / self.num_agent) * np.sqrt(np.sum(np.cos(theta))**2 + np.sum(np.sin(theta))**2)
            
        return grobal_order, local_order, dencity
    
    def generate_position(self):
        x = np.random.uniform(low=0, high=self.lx_out, size=self.num_agent)
        y = np.random.uniform(low=0, high=self.ly_out, size=self.num_agent)
        theta = np.random.uniform(low=-np.pi, high=np.pi, size=self.num_agent)
        #R = np.random.uniform(low=0.25, high=2.0, size=self.num_agent)
        
        #R_1 = np.full(int(self.num_agent / 2), 1.5)
        #R_2 = np.full(int(self.num_agent / 2), 0.5)
        #R = np.hstack((R_1, R_2))
        
        R = np.full(self.num_agent, self.radius)
        
        #R = self.radius
        
        return np.array([x, y, theta, R]).T
    
    def reset(self):
        reset_position = self.generate_position()
        return reset_position
    
    def flag_coll(self, s):
        x, y, _, _ = self.pos_to_vec(s)
        dx = x[np.newaxis, :] - x[:, np.newaxis]
        dy = y[np.newaxis, :] - y[:, np.newaxis]
        dist = np.sqrt(dx**2 + dy**2)
        index_coll = dist < 0.3
        return index_coll
    
    def step(self, s, a):
        past_position = np.array(s)
        next_position = self.get_s_next(s, a)
    
        neighbor_matrix_past = self.cul_neighbor(past_position).sum(axis=1)
        neighbor_matrix_next = self.cul_neighbor(next_position).sum(axis=1)    
        reward = np.where(neighbor_matrix_past > neighbor_matrix_next, 1, 0)
        
        #coll_matrix_past = self.flag_coll(past_position).sum(axis=1)
        #coll_matrix_next = self.flag_coll(next_position).sum(axis=1) 
        #reward = np.where(coll_matrix_past > coll_matrix_next, 0, 1)
        
        #_, reward, _ = self.take_order_and_dencity(s)
            
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
    
    def generate_state(self, position): # position = num_agent x 3
        output_state = self.arg(position)
        
        return output_state # len(output_state) = num_agent
    
    
#%%
#vicsek model の環境を実装
import numpy as np
import random

class Vicsek_model:
    def __init__(self, num_agent, perceptual_range, v, eta, lx_out, ly_out):
        self.N = num_agent
        self.R = perceptual_range
        self.v = v
        self.eta = eta
        self.lx_out = lx_out
        self.ly_out = ly_out
        
    def generator(self, size):
        return np.random.uniform(-1.0, 1.0, size = size)
        
    def pBC(self, x_, l_):
        return (x_ < 0)*(x_ + l_) + (x_ > l_)*(x_ - l_) + (-l_ <= x_ <= l_)*(x_)
    
    def reset(self):
        x = np.random.uniform(high = self.lx_out, low = 0, size = self.N)
        y = np.random.uniform(high = self.ly_out, low = 0, size = self.N)
        angle = np.arctan2(self.generator(self.N), self.generator(self.N))
        return x, y, angle # length = num_agent
    
    # 先行研究におけるstate関数
    def arg(self, x, y, theta):
        dx = x[np.newaxis,:] - x[:,np.newaxis] # num_agent x num_agent
        dy = y[np.newaxis,:] - y[:,np.newaxis]
        dis_cul = dx**2 + dy**2 < self.R**2 
        num_neighbor = dis_cul.sum(axis=1)
        # 方向ベクトル
        v = np.array([np.cos(theta), np.sin(theta)]).T
        #print('shape of v={}'.format(v.shape))
        
        cos = np.cos(theta)
        sin = np.sin(theta)
        cos_neighbor = np.dot(dis_cul, cos)
        sin_neighbor = np.dot(dis_cul, sin)
        cos_mean = np.where(num_neighbor != 0, cos_neighbor / num_neighbor, 0)
        sin_mean = np.where(num_neighbor != 0, sin_neighbor / num_neighbor, 0)
        P = np.array([cos_mean, sin_mean]).T
        P_norm = np.linalg.norm(P, axis=1)
        # 正規化近傍平均方向ベクトル
        P_normalize = P / P_norm[:, np.newaxis]
        #print('shape of p_normalize={}'.format(P_normalize.shape))
        
        # vをπ/2回転させたベクトルとPの内積が正の場合はarccos, その他は-arccosを返す
        theta_r = np.pi/2
        rota = np.array([[np.cos(theta_r), -np.sin(theta_r)],[np.sin(theta_r),  np.cos(theta_r)]])
        rota_v = np.tile(rota, (self.N, 1, 1))
        vTop = v.reshape((self.N, 2, 1))
        rota_dot_vTop = np.matmul(rota_v, vTop).reshape((self.N, 2))
        P_dot_vTop = (P * rota_dot_vTop).sum(axis=1)
        
        # 返り値を計算
        naiseki = (P_normalize * v).sum(axis=1)
        arccos = np.arccos(naiseki)
        output = np.where(P_dot_vTop > 0, arccos, -arccos) # 場合分け
        return output
    
    """
    # 先行研究における数理モデル
    def step(self, x, y, theta):
        state = self.arg(x, y, theta)
        theta_max = (3/16)*np.pi
        
        theta_next = np.where(np.abs(state) <= theta_max, state, np.where(state > theta_max, theta_max, -theta_max))
        theta_next += self.eta * np.random.uniform(low=-1/2, high=1/2, size=self.N)
        
        x_next = x + self.v * np.cos(theta_next) # num_agent x 1
        y_next = y + self.v * np.sin(theta_next)
        
        pBC_v = np.vectorize(self.pBC)
        
        x_next = pBC_v(x_next, self.lx_out)
        y_next = pBC_v(y_next, self.ly_out)
        
        return x_next, y_next, theta_next # length = num_agent : 1 x 1
    """
    
    def step(self, x, y, angle):
        dx = x[np.newaxis,:] - x[:,np.newaxis] # num_agent x num_agent
        dy = y[np.newaxis,:] - y[:,np.newaxis]
        neighbor = dx**2 + dy**2 < self.R**2
        
        cos_angle_matrix = np.tile(np.cos(angle), (self.N, 1)) # num_agent x num_agent
        sin_angle_matrix = np.tile(np.sin(angle), (self.N, 1))
        neighbors_cos = np.where(neighbor == True, cos_angle_matrix, 0.0) # num_agent x num_agent
        neighbors_sin = np.where(neighbor == True, sin_angle_matrix, 0.0)
        cos_mean = np.mean(neighbors_cos, axis = 1) # num_agent x 1
        sin_mean = np.mean(neighbors_sin, axis = 1)
        angle_next = np.arctan2(sin_mean, cos_mean) + np.random.uniform(low=-self.eta*(1/2), high=self.eta*(1/2), size=self.N) # num_agent x 1
        
        x_next = x + self.v * np.cos(angle_next) # num_agent x 1
        y_next = y + self.v * np.sin(angle_next)
        
        pBC_v = np.vectorize(self.pBC)
        
        x_next = pBC_v(x_next, self.lx_out)
        y_next = pBC_v(y_next, self.ly_out)
        
        return x_next, y_next, angle_next # length = num_agent : 1 x 1
    
    def take_parameter(self, x, y, angle):
        dx = x[np.newaxis,:] - x[:,np.newaxis] # num_agent x num_agent
        dy = y[np.newaxis,:] - y[:,np.newaxis]
        neighbor = dx**2 + dy**2 < self.R**2
        
        local_order = []
        density = []
        
        for i in range(self.N):
            neighbor_i = neighbor[i]
            num_neighbor = np.sum(neighbor_i)
            angle_neighbor = angle[neighbor_i == 1]
            cos_neighbor = np.cos(angle_neighbor)
            sin_neighbor = np.sin(angle_neighbor)
            
            density_i = num_neighbor / (np.pi*(self.R**2))
            local_order_i = (1 / num_neighbor) * np.sqrt(np.sum(cos_neighbor)**2 + np.sum(sin_neighbor)**2) if num_neighbor != 0 else 0.0
            density.append(density_i)
            local_order.append(local_order_i)
            
        grobal_order = (1 / self.N) * np.sqrt(np.sum(np.cos(angle))**2 + np.sum(np.sin(angle))**2)
        
        return grobal_order, np.mean(np.array(local_order)), np.mean(np.array(density))


#%% 
#学習過程
import numpy as np
import random
import os
import time

#学習関数
def running_learning_QTable(num):
    num_agent = 100
    #perceptual_range = 1.0
    unit_length = np.round(np.sqrt(num_agent / 20.))
    perceptual_range = unit_length / 3
    lx_out = unit_length
    ly_out = unit_length
    #v0 = 0.5
    #v0 = np.random.uniform(low=0.25, high=0.75, size=num_agent)
    v0 = unit_length / 20
    eta = 0.0
    #eta = np.random.uniform(low=0., high=1.0, size=num_agent)
    episodes = 1000
    Tmax = 10000
    action_size = 7
    state_size = 32
    policy = 'epsilon_greedy'
    tag = str(num + 1)
    
    print('L={}, R={}, v0={}'.format(unit_length, perceptual_range, v0))

    agent = QTableAgent(state_size, action_size, num_agent, policy)
    env = Enviroment_vicsek(action_size, num_agent, perceptual_range, lx_out, ly_out, v0, eta)
    reward_history = []
    reward25_history = []
    reward2_history = []
    grobal_order_history = []
    file_names = []
    
    state_history = []
    action_history = []
    
    for episode in range(episodes):
        positions = env.reset()
        R_dist = positions.T[3]
        total_reward = 0
        grobal_orders = []
        #reward_25 = 0
        #reward_2 = 0
        
        state_record = []
        action_record = []
            
        for t in range(Tmax):
            state = env.generate_state(positions)
            state_record.append(state.reshape((num_agent,)))
            grobal_order, _, _ = env.take_order_and_dencity(positions)
            grobal_orders.append(grobal_order)
                
            actions = agent.get_action(state, episode)
            action_record.append(np.array(actions))
                    
            next_positions, rewards, done = env.step(positions, actions)
            next_state = env.generate_state(next_positions)
                
            agent.update(state, actions, rewards, next_state, done)
                
            positions = next_positions
            total_reward += np.mean(rewards)
            
            #reward_25 += np.mean(rewards[-int(num_agent / 2) :]) 
            #reward_2 += np.mean(rewards[: int(num_agent / 2)])
            
            if t % (Tmax / 100) == 0:
                print("progress={}%".format(int(t/Tmax * 100)), end="\r")
                time.sleep(0.1)
    
        reward_history.append(total_reward)
        #reward25_history.append(reward_25) # 半径0.25
        #reward2_history.append(reward_2) # 半径2.0
        grobal_order_history.append(np.mean(np.array(grobal_orders)))
        state_history.append(np.array(state_record))
        action_history.append(np.array(action_record))
        print('{}th, Episode={}\nTotal Reward={:.3f}, Grobal Order={:.3f}'.format(num + 1, episode + 1, total_reward, np.mean(np.array(grobal_orders))))
        
    
    #学習済みモデルの保存
    folder_name = "sotsuken_MARL_nonuniform"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    # 4. ファイル名の指定
    file_name = 'sotsuken_MARL_nonuniform_sim9' + tag + '.npy'
    file_names.append(file_name)
    file_path = os.path.join(folder_name, file_name)
    QTable = agent.get_QTable()
    np.save(file_path, QTable)
        
    return np.array(reward_history), np.array(file_names) ,np.array(grobal_order_history), np.array(state_history), np.array(action_history), np.array(R_dist), np.array(reward25_history), np.array(reward2_history), eta, v0


#%%
#学習を実行
reward_record = []
file_name_record = []
grobal_order_record = []
state_records = []
action_records = []
R_records = []
reward1_record = []
reward2_record = []
eta_record = []
v_record = []

for i in range(1):
    j = i
    reward_j, file_name_j, grobal_order_j, state_j, action_j, R_j, reward1_j, reward2_j, eta_j, v_j = running_learning_QTable(j)
    reward_record.append(reward_j)
    file_name_record.append(file_name_j)
    grobal_order_record.append(grobal_order_j)
    state_records.append(state_j)
    action_records.append(action_j)
    R_records.append(R_j)
    reward1_record.append(reward1_j)
    reward2_record.append(reward2_j)
    eta_record.append(eta_j)
    v_record.append(v_j)

# sim1, R ~ U(0.25, 2.0)
# sim2, R = 1.0
# sin3, R = 1.0, eta ~ U(0.0, 1.0), Eps=1000
# damn! sim4, reward = 1 if number of collision past > number of collision now else 0, Restricted field of view (3/16)π is damn
# sim5, reward = 0 if number of collision past > number of collision now else 1, Restricted field of view (3/16)π
# sim6, v0 ~ U(0.1, 1.0), その他は均一 damn!!!!
# sim7, 7_2, v0 ~ U(0.25, 0.75), その他は均一


#%%
#データの書き出し
import os
import numpy as np

folder_name = "sotsuken_MARL_nonuniform"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

file_path = os.path.join(folder_name, "data_rewards_sotsuken_MARL_nonuniform_sim9.csv")
data_rewards = np.array(reward_record)
np.savetxt(file_path, data_rewards)

file_path = os.path.join(folder_name, "data_file_names_sotsuken_MARL_nonuniform_sim9.csv")
data_file_names = np.array(file_name_record)
np.savetxt(file_path, data_file_names, fmt='%s')

file_path = os.path.join(folder_name, "data_grobal_order_sotsuken_MARL_nonuniform_sim9.csv")
data_grobal_order = np.array(grobal_order_record)
np.savetxt(file_path, data_grobal_order)

file_path = os.path.join(folder_name, "data_state_sotsuken_MARL_nonuniform_sim1.npy")
#data_state = np.array(state_records)
#np.save(file_path, data_state)

file_path = os.path.join(folder_name, "data_action_sotsuken_MARL_nonuniform_sim1.npy")
#data_action = np.array(action_records)
#np.save(file_path, data_action)

file_path = os.path.join(folder_name, "data_R_sotsuken_MARL_nonuniform_sim9.npy")
data_R = np.array(R_records)
np.save(file_path, data_R)

file_path = os.path.join(folder_name, "data_eta_sotsuken_MARL_nonuniform_sim6.npy")
data_eta = np.array(eta_record)
#np.save(file_path, data_eta)

file_path = os.path.join(folder_name, "data_v_sotsuken_MARL_nonuniform_sim9.npy")
data_v = np.array(v_record)
np.save(file_path, data_v)

file_path = os.path.join(folder_name, "data_reward1_sotsuken_MARL_nonuniform_sim1.csv")
data_reward25 = np.array(reward1_record)
#np.savetxt(file_path, data_reward25)

file_path = os.path.join(folder_name, "data_reward2_sotsuken_MARL_nonuniform_sim1.csv")
data_reward2 = np.array(reward2_record)
#np.savetxt(file_path, data_reward2)


#%%
#学習結果のグラフ作成
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

num_agent = 100
num_clone = num_agent - 1
perceptual_range = 1.0
unit_length = np.round(np.sqrt(num_agent / 2.))
lx_out = unit_length
ly_out = unit_length
v0 = 0.5
eta = 0.0
episodes = 800
Tmax = 10000
sync_interval = 20
action_size = 7
state_size = 32
policy = 'epsilon_greedy'

loaded_rewards = np.loadtxt('pmarl_2d_table_NonUniform_folder/data_rewards_table_NonUniform_sim5.csv')
loaded_rewards_mean = np.mean(loaded_rewards, axis=0)

loaded_reward1 = np.loadtxt('pmarl_2d_table_NonUniform_folder/data_reward1_table_NonUniform_sim5.csv')
loaded_reward25_mean = np.mean(loaded_reward1, axis=0)

loaded_reward2 = np.loadtxt('pmarl_2d_table_NonUniform_folder/data_reward2_table_NonUniform_sim5.csv')
loaded_reward2_mean = np.mean(loaded_reward2, axis=0)

loaded_grobal_order = np.loadtxt('pmarl_2d_table_NonUniform_folder/data_grobal_order_table_NonUniform_sim5.csv')
loaded_grobal_order_mean = np.mean(loaded_grobal_order, axis=0)

data_R = np.load('pmarl_2d_table_NonUniform_folder/data_R_table_NonUniform_sim5.npy')

fig8 = plt.figure(figsize = (30, 30))
ax = fig8.add_subplot(311)
ax.grid(True)
ax.set_yticks(np.arange(0, 11000, step=1000))
ax.set_xlim(0, episodes)
ax.set_ylim(0, 10000)
ax.set_title(r'Changes in Total Reward (= Local Order) in learning, Episodes={}, System size=$[0, {}]^2$, Density in the system={:.3f}'.format(episodes, lx_out, num_agent / (lx_out * ly_out), ) + '\nNumber of agents={}, Time steps={}, Number of trials={}, mean in t > {} is {:.3f}'.format(num_agent, Tmax, 20, int(episodes/2), np.mean(loaded_rewards_mean[int(episodes/2):])), fontsize=14)
ax.set_xlabel('Episode', fontsize=14)
ax.set_ylabel('Total Reward', fontsize=14)

#line, = ax.plot(np.arange(len(loaded_rewards_mean)), loaded_rewards_mean, color = 'k')
for i in range(len(loaded_rewards)):
    if i == (len(loaded_rewards) - 1):
        ax.plot(np.arange(len(loaded_rewards[i])), loaded_rewards[i], color='green', alpha=0.5, label='All agents')
        ax.plot(np.arange(len(loaded_reward1[i])), loaded_reward1[i], color='red', alpha=0.5, label='R = 1.5')
        ax.plot(np.arange(len(loaded_reward2[i])), loaded_reward2[i], color='blue', alpha=0.5, label='R = 0.5') 
    ax.plot(np.arange(len(loaded_rewards[i])), loaded_rewards[i], color='green', alpha=0.5)
    ax.plot(np.arange(len(loaded_reward1[i])), loaded_reward1[i], color='red', alpha=0.5)
    ax.plot(np.arange(len(loaded_reward2[i])), loaded_reward2[i], color='blue', alpha=0.5)
ax.legend(loc='best', fontsize=18)


ax2 = fig8.add_subplot(312)
ax2.grid(True)
ax2.set_yticks(np.arange(0, 1.1, step=0.1))
ax2.set_xlim(0, episodes)
ax2.set_ylim(0, 1)
ax2.set_title(r'Changes in Grobal Order in learning, Episodes={}, System size=$[0, {}]^2$, Density in the system={:.3f}'.format(episodes, lx_out, num_agent / (lx_out * ly_out), ) + '\nNumber of agents={}, Time steps={}, Number of trials={}, mean in t > {} is {:.3f}'.format(num_agent, Tmax, 20, int(episodes/2), np.mean(loaded_grobal_order_mean[int(episodes/2):])), fontsize=14)
ax2.set_xlabel('Episode', fontsize=14)
ax2.set_ylabel('Grobal Order', fontsize=14)

#line2, = ax2.plot(np.arange(len(loaded_grobal_order_mean)), loaded_grobal_order_mean, color = 'k')
for i in range(len(loaded_grobal_order)):
    ax2.plot(np.arange(len(loaded_grobal_order[i])), loaded_grobal_order[i], color='green', alpha=0.5)
#ax2.legend([line2], ['Average of trials'], loc='lower right', fontsize=18)

ax3 = fig8.add_subplot(313)
ax3.grid(True)
#ax3.set_yticks(np.arange(0, 1.1, step=0.1))
#ax3.set_xlim(0, episodes)
#ax3.set_ylim(0, 1)
ax3.set_title(r'Distribution of perceptual range in learning, Episodes={}, System size=$[0, {}]^2$, Density in the system={:.3f}'.format(episodes, lx_out, num_agent / (lx_out * ly_out), ) + '\nNumber of agents={}, Time steps={}, Number of trials={}, mean in t > {} is {:.3f}'.format(num_agent, Tmax, 20, int(episodes/2), np.mean(loaded_grobal_order_mean[int(episodes/2):])), fontsize=14)
ax3.set_xlabel('Perceptual range', fontsize=14)
ax3.set_ylabel('Number of agents', fontsize=14)
bin = int(1 + np.log2(num_agent))
for i in range(len(data_R)):
    sns.histplot(data = data_R[i], alpha = 0.3, kde = True, ax=ax3)

plt.show()

folder_name = "pmarl_2d_table_NonUniform_folder"
plt.savefig(os.path.join(folder_name, 'pmarl_2d_table_NonUniform_trials_sim5.png'), dpi=300)



#%%
# UniformとNonUniformで比較
#学習結果のグラフ作成
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

num_agent = 100
num_clone = num_agent - 1
perceptual_range = 1.0
unit_length = np.round(np.sqrt(num_agent / 2.))
lx_out = unit_length
ly_out = unit_length
v0 = 0.5
eta = 0.0
episodes = 800
Tmax = 10000
sync_interval = 20
action_size = 7
state_size = 32
policy = 'epsilon_greedy'

# Nonuniform data
loaded_rewards_non = np.loadtxt('pmarl_2d_table_NonUniform_folder/data_rewards_table_NonUniform_sim1.csv')
loaded_rewards_mean_non = np.mean(loaded_rewards, axis=0)

loaded_grobal_order_non = np.loadtxt('pmarl_2d_table_NonUniform_folder/data_grobal_order_table_NonUniform_sim1.csv')
loaded_grobal_order_mean_non = np.mean(loaded_grobal_order_non, axis=0)

# Uniform data
loaded_rewards = np.loadtxt('pmarl_2d_table_folder/data_rewards_table_sim9.csv')
loaded_rewards_mean = np.mean(loaded_rewards, axis=0)

loaded_grobal_order = np.loadtxt('pmarl_2d_table_folder/data_grobal_order_table_sim9.csv')
loaded_grobal_order_mean = np.mean(loaded_grobal_order, axis=0)


fig11 = plt.figure(figsize = (30, 30))
ax = fig11.add_subplot(211)
ax.grid(True)
ax.set_yticks(np.arange(0, 11000, step=1000))
ax.set_xlim(0, episodes)
ax.set_ylim(0, 10000)
title1 = r'Changes in Reward in learning, Episodes={}, System size=$[0, {}]^2$, Density in the system={:.3f}'.format(episodes, lx_out, num_agent / (lx_out * ly_out)) 
title2 = '\nUniform : Number of agents={}, Time steps={}, Number of trials={}, mean in t > {} is {:.3f}'.format(num_agent, Tmax, 20, int(episodes/2), np.mean(loaded_rewards_mean[int(episodes/2):]))
title3 = '\nNon Uniform : Number of agents={}, Time steps={}, Number of trials={}, mean in t > {} is {:.3f}'.format(num_agent, Tmax, 20, int(episodes/2), np.mean(loaded_rewards_mean_non[int(episodes/2):]))
ax.set_title(title1 + title2 + title3, fontsize=20)
ax.set_xlabel('Episode', fontsize=20)
ax.set_ylabel('Total Reward', fontsize=20)

#line, = ax.plot(np.arange(len(loaded_rewards_mean)), loaded_rewards_mean, color = 'k')
for i in range(len(loaded_rewards)):
    ax.plot(np.arange(len(loaded_rewards[i])), loaded_rewards[i], color='green', alpha=0.5)
    ax.plot(np.arange(len(loaded_rewards_non[i])), loaded_rewards_non[i], color='red', alpha=0.5)
    if i == len(loaded_rewards) - 1:
        ax.plot(np.arange(len(loaded_rewards[i])), loaded_rewards[i], color='green', alpha=0.5, label='Uniform')
        ax.plot(np.arange(len(loaded_rewards_non[i])), loaded_rewards_non[i], color='red', alpha=0.5, label='Non Uniform')
ax.legend(loc='best', fontsize=30)
#ax.legend([line], ['Average of trials'], loc='lower right', fontsize=18)


ax2 = fig11.add_subplot(212)
ax2.grid(True)
ax2.set_yticks(np.arange(0, 1.1, step=0.1))
ax2.set_xlim(0, episodes)
ax2.set_ylim(0, 1)
title1 = r'Changes in Grobal Order in learning, Episodes={}, System size=$[0, {}]^2$, Density in the system={:.3f}'.format(episodes, lx_out, num_agent / (lx_out * ly_out)) 
title2 = '\nUniform : Number of agents={}, Time steps={}, Number of trials={}, mean in t > {} is {:.3f}'.format(num_agent, Tmax, 20, int(episodes/2), np.mean(loaded_grobal_order_mean[int(episodes/2):]))
title3 = '\nNon Uniform : Number of agents={}, Time steps={}, Number of trials={}, mean in t > {} is {:.3f}'.format(num_agent, Tmax, 20, int(episodes/2), np.mean(loaded_grobal_order_mean_non[int(episodes/2):]))
ax2.set_title(title1 + title2 + title3, fontsize=20)
ax2.set_xlabel('Episode', fontsize=20)
ax2.set_ylabel('Grobal Order', fontsize=20)

#line2, = ax2.plot(np.arange(len(loaded_grobal_order_mean)), loaded_grobal_order_mean, color = 'k')
for i in range(len(loaded_grobal_order)):
    ax2.plot(np.arange(len(loaded_grobal_order[i])), loaded_grobal_order[i], color='green', alpha=0.5)
    ax2.plot(np.arange(len(loaded_grobal_order_non[i])), loaded_grobal_order_non[i], color='red', alpha=0.5)
    if i == len(loaded_grobal_order) - 1:
        ax2.plot(np.arange(len(loaded_grobal_order[i])), loaded_grobal_order[i], color='green', alpha=0.5, label='Uniform')
        ax2.plot(np.arange(len(loaded_grobal_order_non[i])), loaded_grobal_order_non[i], color='red', alpha=0.5, label='Non Uniform')
ax2.legend(loc='lower right', fontsize=30)

plt.show()

folder_name = "pmarl_2d_table_NonUniform_folder"
plt.savefig(os.path.join(folder_name, 'pmarl_2d_table_NonUniform_vs_Uniform.png'), dpi=300)


#%% 
#学習済みエージェントによるシミュレーションをnum_test回繰り返して平均を取る
import numpy as np

#実行関数の定義
def running_simulation(num, num_test, agent):
    reward_record_mean = []
    local_order_record_mean = []
    grobal_order_record_mean = []
    
    for j in range(num_test):
        positions = env.reset()
    
        reward_records = []
        local_order_record = []
        grobal_order_record = []
        
        state_record = []
        action_record = []
    
        for t in range(Tmax):
            state = env.generate_state(positions)
            
            grobal_order, local_orders, _ = env.take_order_and_dencity(positions)
            grobal_order_record.append(grobal_order)
            local_order_record.append(np.mean(local_orders))
            
            actions = agent.get_action(state, episodes)
                    
            next_positions, rewards, done = env.step(positions, actions)
            reward_records.append(np.mean(rewards))
        
            positions = next_positions
        
        reward_record_mean.append(np.array(reward_records))
        local_order_record_mean.append(np.array(local_order_record))
        grobal_order_record_mean.append(np.array(grobal_order_record))
        
        progress_rate = (j+1)/num_test*100
        print('{}th Progress={:.1f}%'.format(num + 1, progress_rate), end='\r')
        
    return reward_record_mean, local_order_record_mean, grobal_order_record_mean


#%%
#繰り返し実行の平均をとる
import numpy as np

#各種定数
num_agent = 100
num_clone = num_agent - 1
perceptual_range = 1.0
unit_length = np.round(np.sqrt(num_agent / 2.))
lx_out = unit_length
ly_out = unit_length
v0 = 0.5
eta = 0.0
episodes = 800
Tmax = 1000
sync_interval = 20
action_size = 7
state_size = 32
policy = 'greedy'
num_test = 20

#ファイル名を取得
#file_name_data = np.loadtxt('sotsuken_MARL_nonuniform/data_file_names_sotsuken_MARL_nonuniform_sim3.csv', delimiter=',', dtype='str')
#data_R = np.load('sotsuken_MARL_nonuniform/data_R_sotsuken_MARL_nonuniform_sim3.npy')
#data_eta = np.load('sotsuken_MARL_nonuniform/data_eta_sotsuken_MARL_nonuniform_sim3.npy')
data_v = np.load('sotsuken_MARL_nonuniform/data_v_sotsuken_MARL_nonuniform_sim7.npy')

reward_record_means = []
local_order_record_means = []
grobal_order_record_means = []
density_record_means = []

for i in range(1):
    #エージェントの定義
    policy = 'greedy'
    env = Enviroment_vicsek(action_size, num_agent, perceptual_range, lx_out, ly_out, data_v[0], eta)
    agent = QTableAgent(state_size, action_size, num_agent, policy)
    
    # 1. モデルの保存ディレクトリとファイル名を指定する
    #model_filename = file_name_data

    # 2. モデルを呼び出す
    loaded_model_state = agent.load_q_table('sotsuken_MARL_nonuniform/sotsuken_MARL_nonuniform_sim71.npy')

    # 3. シミュレーションを実行
    reward_record_mean, local_order_record_mean, grobal_order_record_mean = running_simulation(i, num_test, agent)
    reward_record_means.append(reward_record_mean)
    local_order_record_means.append(local_order_record_mean)
    grobal_order_record_means.append(grobal_order_record_mean)
    

#%%
#ファイルの保存
import os
import numpy as np

folder_name = "sotsuken_MARL_nonuniform"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

file_path = os.path.join(folder_name, "data_rrm_sim7.npy")
data_rewards = np.array(reward_record_means)
np.save(file_path, data_rewards)

file_path = os.path.join(folder_name, "data_lorm_sim7.npy")
data_file_names = np.array(local_order_record_means)
np.save(file_path, data_file_names)

file_path = os.path.join(folder_name, "data_gorm_sim7.npy")
data_file_names = np.array(grobal_order_record_means)
np.save(file_path, data_file_names)


#%%
#学習済みエージェントによる20回のシミュレーションにおいて得られた報酬の推移の平均を描画
import numpy as np
import matplotlib.pyplot as plt
import os

#各種変数
num_agent = 100
num_clone = num_agent - 1
perceptual_range = 1.0
unit_length = np.round(np.sqrt(num_agent / 2.))
lx_out = unit_length
ly_out = unit_length
v0 = 0.5
eta = 0.0
episodes = 600
Tmax = 500
sync_interval = 20
action_size = 7
state_size = 32
policy = 'greedy'
num_test = 20

#フォルダの作成
folder_name = "pmarl_2d_table_NonUniform_folder"
os.makedirs(folder_name, exist_ok=True)

#データのロード
loaded_rewards = np.loadtxt('pmarl_2d_table_NonUniform_folder/data_rewards_table_NonUniform_sim1.csv')
file_name_data = np.loadtxt('pmarl_2d_table_NonUniform_folder/data_file_names_table_NonUniform_sim1.csv', delimiter=',', dtype='str')
data_rrm = np.load("pmarl_2d_table_NonUniform_folder/data_rrm_table_NonUniform_sim1.npy")
data_lorm = np.load("pmarl_2d_table_NonUniform_folder/data_lorm_table_NonUniform_sim1.npy")
data_grom = np.load("pmarl_2d_table_NonUniform_folder/data_gorm_table_NonUniform_sim1.npy")
data_drm = np.load("pmarl_2d_table_NonUniform_folder/data_drm_table_NonUniform_sim1.npy")

#平均のリスト
means_grobal = []
means_local = []
means_drm = []

#画像の作成＆フォルダへの保存
for k in range(len(file_name_data)):
    #ファイル名の作成
    tag = 'pmarl_2d_table_NonUniform_sim1_' + str(k+1) + '.png'
    
    fig3 = plt.figure(figsize = (16,48))

    RRM = np.array(data_rrm[k])
    RRM_mean = np.mean(RRM, axis=0)

    LORM = np.array(data_lorm[k])
    LORM_mean = np.mean(LORM, axis=0)
    means_local.append(LORM_mean)

    GORM = np.array(data_grom[k])
    GORM_mean = np.mean(GORM, axis=0)
    means_grobal.append(GORM_mean)

    DRM = np.array(data_drm[k])
    DRM_mean = np.mean(DRM, axis=0)
    means_drm.append(DRM_mean)
    
    ax5 = fig3.add_subplot(511)
    ax5.grid(True)
    ax5.set_yticks(np.arange(0, 11000, step=1000))
    ax5.set_xlim(0, episodes)
    ax5.set_ylim(0, Tmax)
    ax5.set_title(r'Changes in Total Reward in learning, Episodes={}, System size=$[0, {}]^2$, Density in the system={:.3f}'.format(episodes, lx_out, num_agent / (lx_out * ly_out)) + '\nNumber of agents={}, Tmax={}, Number of times learned={}'.format(num_agent, Tmax, 10, fontsize=14))
    ax5.set_xlabel('time', fontsize=14)
    ax5.set_ylabel('Total Reward', fontsize=14)

    line_k, = ax5.plot(np.arange(len(loaded_rewards[k])), loaded_rewards[k], color = 'k')
    ax5.legend([line_k], ['Total reward'], loc='lower right', fontsize=18)

    ax = fig3.add_subplot(512)
    ax.grid(True)
    ax.set_yticks(np.arange(0, 1.1, step=0.1))
    ax.set_xlim(0, Tmax)
    ax.set_ylim(0, 1)
    ax.set_title(r'Changes in mean of Reward, System size=$[0, {}]^2$, Density in the system={:.3f}'.format(lx_out, num_agent / (lx_out * ly_out)) + '\nN={}, Time steps={}, Number of trial={}, mean in t > {} is {:.3f}'.format(num_agent, Tmax, num_test, int(Tmax/2), np.mean(RRM_mean[int(Tmax/2):])), fontsize=14)
    ax.set_xlabel('time', fontsize=14)
    ax.set_ylabel('Reward', fontsize=14)

    line, = ax.plot(np.arange(len(RRM_mean)), RRM_mean, color = 'k')
    for i in range(num_test):
        ax.plot(np.arange(len(RRM[i])), RRM[i], color='green', alpha=0.1)
    ax.legend([line], ['mean of reward'], loc='lower right', fontsize=18)

    ax2 = fig3.add_subplot(513)
    ax2.grid(True)
    ax2.set_yticks(np.arange(0, 1.1, step=0.1))
    ax2.set_xlim(0, Tmax)
    ax2.set_ylim(0, 1)
    ax2.set_title(r'Changes in mean of local order, System size=$[0, {}]^2$, Density in the system={:.3f}'.format(lx_out, num_agent / (lx_out * ly_out)) + '\nN={}, Time steps={}, Number of trial={}, mean in t > {} is {:.3f}'.format(num_agent, Tmax, num_test, int(Tmax/2), np.mean(LORM_mean[int(Tmax/2):])), fontsize=14)
    ax2.set_xlabel('time', fontsize=14)
    ax2.set_ylabel('Local order \n local order is an order parameter within the local radius of each agent.', fontsize=14)

    line2, = ax2.plot(np.arange(len(LORM_mean)), LORM_mean, color = 'k')
    for i in range(num_test):
        ax2.plot(np.arange(len(LORM[i])), LORM[i], color='green', alpha=0.1)
    ax2.legend([line2], ['mean of local order'], loc='lower right', fontsize=18)

    ax4 = fig3.add_subplot(514)
    ax4.grid(True)
    ax4.set_yticks(np.arange(0, 1.1, step=0.1))
    ax4.set_xlim(0, Tmax)
    ax4.set_ylim(0, 1)
    ax4.set_title(r'Changes in mean of grobal order, System size=$[0, {}]^2$, Density in the system={:.3f}'.format(lx_out, num_agent / (lx_out * ly_out)) + '\nN={}, Time steps={}, Number of trial={}, mean in t > {} is {:.3f}'.format(num_agent, Tmax, num_test, int(Tmax/2), np.mean(GORM_mean[int(Tmax/2):])), fontsize=14)
    ax4.set_xlabel('time', fontsize=14)
    ax4.set_ylabel('Grobal order \n grobal order is an order parameter for the entire system.', fontsize=14)

    line4, = ax4.plot(np.arange(len(GORM_mean)), GORM_mean, color = 'k')
    for i in range(num_test):
        ax4.plot(np.arange(len(GORM[i])), GORM[i], color='green', alpha=0.1)
    ax4.legend([line4], ['mean of grobal order'], loc='lower right', fontsize=18)

    ax3 = fig3.add_subplot(515)
    ax3.grid(True)
    ax3.set_xlim(0, Tmax)
    #ax3.set_ylim(0, 1.0)
    ax3.set_title(r'Changes in mean of Number of neighbors, System size=$[0, {}]^2$, Density in the system={:.3f}'.format(lx_out, num_agent / (lx_out * ly_out)) + '\nN={}, Time steps={}, Number of trial={}'.format(num_agent, Tmax, num_test), fontsize=14)
    ax3.set_xlabel('time', fontsize=14)
    ax3.set_ylabel('density within a perceptual range={}'.format(perceptual_range), fontsize=14)

    line3, = ax3.plot(np.arange(len(DRM_mean)), DRM_mean, color = 'k')
    for i in range(num_test):
        ax3.plot(np.arange(len(DRM[i])), DRM[i], color='green', alpha=0.1)
    ax3.legend([line3], ['mean of density'], loc='best', fontsize=18)
    
    #画像をフォルダに保存
    plt.savefig(os.path.join(folder_name, tag), dpi=300)
    

#%%
#個別に学習した20体のエージェントをそれぞれ20回ずつシミュレーションさせ、各パラメータの平均のグラフを重ね書きする
import numpy as np
import matplotlib.pyplot as plt

#各種変数
num_agent = 100
num_clone = num_agent - 1
perceptual_range = 1.0
unit_length = np.round(np.sqrt(num_agent / 2.))
lx_out = unit_length
ly_out = unit_length
v0 = 0.5
eta = 0.0
episodes = 600
Tmax = 500
sync_interval = 20
action_size = 7
state_size = 32
policy = 'greedy'
num_test = 20

#フォルダの作成
folder_name = "pmarl_2d_table_NonUniform_folder"
os.makedirs(folder_name, exist_ok=True)

#データのロード
loaded_rewards = np.loadtxt('pmarl_2d_table_NonUniform_folder/data_rewards_table_NonUniform_sim1.csv')
file_name_data = np.loadtxt('pmarl_2d_table_NonUniform_folder/data_file_names_table_NonUniform_sim1.csv', delimiter=',', dtype='str')
data_rrm = np.load("pmarl_2d_table_NonUniform_folder/data_rrm_table_NonUniform_sim1.npy")
data_lorm = np.load("pmarl_2d_table_NonUniform_folder/data_lorm_table_NonUniform_sim1.npy")
data_grom = np.load("pmarl_2d_table_NonUniform_folder/data_gorm_table_NonUniform_sim1.npy")
data_drm = np.load("pmarl_2d_table_NonUniform_folder/data_drm_table_NonUniform_sim1.npy")

#平均のリスト
means_grobal = []
means_local = []
means_drm = []

#画像の作成＆フォルダへの保存
for k in range(len(file_name_data)):
    RRM = np.array(data_rrm[k])
    RRM_mean = np.mean(RRM, axis=0)

    LORM = np.array(data_lorm[k])
    LORM_mean = np.mean(LORM, axis=0)
    means_local.append(LORM_mean)

    GORM = np.array(data_grom[k])
    GORM_mean = np.mean(GORM, axis=0)
    means_grobal.append(GORM_mean)

    DRM = np.array(data_drm[k])
    DRM_mean = np.mean(DRM, axis=0)
    means_drm.append(DRM_mean)

means_local_array = np.array(means_local)
averages_local = np.mean(means_local_array, axis=0)

means_grobal_array = np.array(means_grobal)
averages_grobal = np.mean(means_grobal_array, axis=0)

means_drm_array = np.array(means_drm)
averages_drm = np.mean(means_drm_array, axis=0)

fig9 = plt.figure(figsize = (16,24))

ax = fig9.add_subplot(311)
ax.grid(True)
ax.set_yticks(np.arange(0, 1.1, step=0.1))
ax.set_xlim(0, Tmax)
ax.set_ylim(0, 1)
ax.set_title(r'Changes in mean of local order, System size=$[0, {}]^2$, Density in the system={:.3f}'.format(lx_out, num_agent / (lx_out * ly_out)) + '\nN={}, Time steps={}, Number of trial={}, mean in t > {} is {:.3f}'.format(num_agent, Tmax, num_test, int(Tmax/2), np.mean(averages_local[int(Tmax/2):])), fontsize=14)
ax.set_xlabel('time', fontsize=14)
ax.set_ylabel('grobal order \n grobal order is an order parameter for the entire system.', fontsize=14)

line, = ax.plot(np.arange(len(averages_local)), averages_local, color = 'k')
for i in range(len(means_local_array)):
    ax.plot(np.arange(len(means_local_array[i])), means_local_array[i], color='green', alpha=0.3)
ax.legend([line], ['mean of local order'], loc='lower right', fontsize=18)

ax2 = fig9.add_subplot(312)
ax2.grid(True)
ax2.set_yticks(np.arange(0, 1.1, step=0.1))
ax2.set_xlim(0, Tmax)
ax2.set_ylim(0, 1)
ax2.set_title(r'Changes in mean of grobal order, System size=$[0, {}]^3$, Density in the system={:.3f}'.format(lx_out, num_agent / (lx_out * ly_out)) + '\nN={}, Time steps={}, Number of trial={}, mean in t > {} is {:.3f}'.format(num_agent, Tmax, num_test, int(Tmax/2), np.mean(averages_grobal[int(Tmax/2):])), fontsize=14)
ax2.set_xlabel('time', fontsize=14)
ax2.set_ylabel('grobal order \n grobal order is an order parameter for the entire system.', fontsize=14)

line2, = ax2.plot(np.arange(len(averages_grobal)), averages_grobal, color = 'k')
for i in range(len(means_grobal_array)):
    ax2.plot(np.arange(len(means_grobal_array[i])), means_grobal_array[i], color='green', alpha=0.3)
ax2.legend([line2], ['mean of grobal order'], loc='lower right', fontsize=18)

ax3 = fig9.add_subplot(313)
ax3.grid(True)
#ax3.set_yticks(np.arange(0, 1.1, step=0.1))
ax3.set_xlim(0, Tmax)
#ax3.set_ylim(0, 1)
ax3.set_title(r'Changes in mean of Number of neighbors, System size=$[0, {}]^2$, Density in the system={:.3f}'.format(10, num_agent / (lx_out * ly_out)) + '\nN={}, Time steps={}, Number of trial={}, mean in t > {} is {:.3f}'.format(num_agent, Tmax, num_test, int(Tmax/2), np.mean(averages_grobal[int(Tmax/2):])), fontsize=14)
ax3.set_xlabel('time', fontsize=14)
ax3.set_ylabel('Number of agents within a perceptual range={}'.format(perceptual_range), fontsize=14)

line3, = ax3.plot(np.arange(len(averages_drm)), averages_drm, color = 'k')
for i in range(len(means_drm_array)):
    ax3.plot(np.arange(len(means_drm_array[i])), means_drm_array[i], color='green', alpha=0.3)
ax3.legend([line3], ['mean of density'], loc='lower right', fontsize=18)

plt.savefig(os.path.join(folder_name, 'pmarl_2d_table_NonUniform_TmaxtimesSim_sim1.png'), dpi=300)


#%%
data_R = np.load('sotsuken_MARL_nonuniform/data_R_sotsuken_MARL_nonuniform_sim3.npy')
data_eta = np.load('sotsuken_MARL_nonuniform/data_eta_sotsuken_MARL_nonuniform_sim3.npy')
print('data_R={}'.format(data_R))
print('data_eta={}'.format(data_eta))

#%%    
# order - etaグラフ作成用シミュレーション

#各種定数
num_agent = 100
num_clone = num_agent - 1
perceptual_range = 1.0
unit_length = np.round(np.sqrt(num_agent / 2.))
lx_out = unit_length
ly_out = unit_length
v0 = 0.5
eta = 0.0
episodes = 600
Tmax = 2000
sync_interval = 20
action_size = 7
state_size = 32
policy = 'greedy'
num_test = 20

#ファイル名を取得
#file_name_data = np.loadtxt('pmarl_2d_table_NonUniform_folder/data_file_names_table_NonUniform_sim1.csv', delimiter=',', dtype='str')

agent = QTableAgent(state_size, action_size, num_agent, policy)
data_R = np.load('sotsuken_MARL_nonuniform/data_R_sotsuken_MARL_nonuniform_sim3.npy')
#data_eta = np.load('sotsuken_MARL_nonuniform/data_eta_sotsuken_MARL_nonuniform_sim3.npy')

# 1. モデルの保存ディレクトリとファイル名を指定する
#model_filename = file_name_data[0]

# 2. モデルを読み込む
#agent.load_q_table('sotsuken_MARL_nonuniform/sotsuken_MARL_nonuniform_sim11.npy')

def simulate_order_eta(agent):
    changes_grobal_order_record = []
    changes_local_order_record = []
    changes_grobal_order_record_vicsek = []
    changes_local_order_record_vicsek = [] 
    
    cost_record = []

    for j in range(11):
        changes_grobal_order_study = []
        changes_local_order_study = []
        changes_grobal_order_vicsek = []
        changes_local_order_vicsek = []
        
        changes_cost = []
        
        eta_add = j * 0.4
        env = Enviroment_vicsek(action_size, num_agent, data_R[0], lx_out, ly_out, v0, eta + np.full(num_agent, eta_add))
        env_vicsek = Vicsek_model(num_agent, perceptual_range, v0, eta, lx_out, ly_out)
        
        positions = env.reset()
        x_vicsek, y_vicsek, theta_vicsek = env_vicsek.reset()
        
        for t in range(Tmax):
            grobal_order, local_order, _ = env.take_order_and_dencity(positions)
            changes_grobal_order_study.append(grobal_order)
            changes_local_order_study.append(np.mean(local_order))
            
            state = env.generate_state(positions)
            
            actions = np.array(agent.get_action(state, episodes))
                
            next_positions, cost, done = env.step(positions, actions)
            
            positions = next_positions
            
            changes_cost.append(np.mean(cost))
            
            #以下はvicsek modelのシミュレーションループ
            grobal_order_vicsek, local_order_vicsek, density_vicsek = env_vicsek.take_parameter(x_vicsek, y_vicsek, theta_vicsek)
            next_x, next_y, next_theta = env_vicsek.step(x_vicsek, y_vicsek, theta_vicsek)
            
            x_vicsek, y_vicsek, theta_vicsek = next_x, next_y, next_theta
            
            changes_grobal_order_vicsek.append(grobal_order_vicsek)
            changes_local_order_vicsek.append(local_order_vicsek)
            
        changes_grobal_order_record.append(np.array(changes_grobal_order_study))
        changes_local_order_record.append(np.array(changes_local_order_study))
        cost_record.append(np.array(changes_cost))
        
        changes_grobal_order_record_vicsek.append(np.array(changes_grobal_order_vicsek))
        changes_local_order_record_vicsek.append(np.array(changes_local_order_vicsek))
        
        progress_rate = (j+1)/11*100
        print(f"Progress: {progress_rate:.1f}%", end='\r')
        
    return changes_grobal_order_record, changes_local_order_record, cost_record


#%%
#ファイルの保存
import os
import numpy as np

agent = QTableAgent(state_size, action_size, num_agent, policy)
agent.load_q_table('sotsuken_MARL_nonuniform/sotsuken_MARL_nonuniform_sim11.npy')
cgor, clor, cc = simulate_order_eta(agent)

folder_name = "sotsuken_MARL_nonuniform"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

file_path = os.path.join(folder_name, "data_cgor_sim1_5.npy")
data_rewards = np.array(cgor)
np.save(file_path, data_rewards)

file_path = os.path.join(folder_name, "data_clor_sim1_5.npy")
data_file_names = np.array(clor)
np.save(file_path, data_file_names)

file_path = os.path.join(folder_name, "data_cc_sim1_5.npy")
data_file_names = np.array(cc)
np.save(file_path, data_file_names)


#%%    
# Uniform vs Non Uniform, order - etaグラフ作成用シミュレーション

#各種定数
num_agent = 100
num_clone = num_agent - 1
perceptual_range = 1.0
unit_length = np.round(np.sqrt(num_agent / 2.))
lx_out = unit_length
ly_out = unit_length
v0 = 0.5
eta = 0.0
episodes = 600
Tmax = 2000
sync_interval = 20
action_size = 7
state_size = 32
policy = 'greedy'
num_test = 20

#ファイル名を取得
file_name_data_non = np.loadtxt('pmarl_2d_table_NonUniform_folder/data_file_names_table_NonUniform_sim1.csv', delimiter=',', dtype='str')
file_name_data = np.loadtxt('pmarl_2d_table_folder/data_file_names_table_sim9.csv', delimiter=',', dtype='str')

agent_uni = QTableAgent(state_size, action_size, num_agent, policy)
agent = QTableAgent(state_size, action_size, num_agent, policy)

# 1. モデルの保存ディレクトリとファイル名を指定する
model_filename_non = file_name_data_non[0]
model_filename = file_name_data[0]

# 2. モデルを読み込む
agent.load_q_table('pmarl_2d_table_NonUniform_folder/' + model_filename_non)
agent_uni.load_q_table('pmarl_2d_table_folder/' + model_filename)

changes_grobal_order_record_uni = []
changes_local_order_record_uni = []
changes_density_record_uni = []

changes_grobal_order_record = []
changes_local_order_record = []
changes_density_record = []

changes_grobal_order_record_vicsek = []
changes_local_order_record_vicsek = []
changes_density_record_vicsek = []

for j in range(11):
    changes_grobal_order_study_uni = []
    changes_local_order_study_uni = []
    changes_density_study_uni = []
    
    changes_grobal_order_study = []
    changes_local_order_study = []
    changes_density_study = []
    
    changes_grobal_order_vicsek = []
    changes_local_order_vicsek = []
    changes_density_vicsek = []
    
    eta = j * 0.4
    env = Enviroment_vicsek(action_size, num_agent, perceptual_range, lx_out, ly_out, v0, eta)
    env_unfiform = Enviroment_vicsek_uniform(action_size, num_agent, perceptual_range, lx_out, ly_out, v0, eta)
    env_vicsek = Vicsek_model(num_agent, perceptual_range, v0, eta, lx_out, ly_out)
    
    positions = env.reset()
    positions_uni = env_unfiform.reset()
    x_vicsek, y_vicsek, theta_vicsek = env_vicsek.reset()
    
    for t in range(Tmax):
        # Non Uniform
        x_vec, y_vec, angle_vec, R = env.pos_to_vec(positions)
        
        grobal_order, local_order, density = env.take_order_and_dencity(positions)
        changes_grobal_order_study.append(grobal_order)
        changes_local_order_study.append(np.mean(local_order))
        changes_density_study.append(np.mean(density))
        
        state = env.generate_state(positions)
        
        actions = np.array(agent.get_action(state, episodes))
            
        next_positions, rewards, done = env.step(positions, actions)
        
        positions = next_positions
        
        # Uniform
        x_uni, y_uni, angle_uni = env_unfiform.pos_to_vec(positions)
        
        grobal_order_uni, local_order_uni, density_uni = env_unfiform.take_order_and_dencity(positions)
        changes_grobal_order_study_uni.append(grobal_order_uni)
        changes_local_order_study_uni.append(np.mean(local_order_uni))
        changes_density_study_uni.append(np.mean(density_uni))
        
        state_uni = env_unfiform.generate_state(positions_uni)
        
        actions_uni = np.array(agent_uni.get_action(state_uni, episodes))
            
        next_positions_uni, rewards_uni, done = env_unfiform.step(positions_uni, actions_uni)
        
        positions_uni = next_positions_uni
        
        #以下はvicsek modelのシミュレーションループ
        grobal_order_vicsek, local_order_vicsek, density_vicsek = env_vicsek.take_parameter(x_vicsek, y_vicsek, theta_vicsek)
        next_x, next_y, next_theta = env_vicsek.step(x_vicsek, y_vicsek, theta_vicsek)
        
        x_vicsek, y_vicsek, theta_vicsek = next_x, next_y, next_theta
        
        changes_grobal_order_vicsek.append(grobal_order_vicsek)
        changes_local_order_vicsek.append(local_order_vicsek)
        changes_density_vicsek.append(density_vicsek)
        
    changes_grobal_order_record.append(np.array(changes_grobal_order_study))
    changes_local_order_record.append(np.array(changes_local_order_study))
    changes_density_record.append(np.array(changes_density_study))
    
    changes_grobal_order_record_uni.append(np.array(changes_grobal_order_study_uni))
    changes_local_order_record_uni.append(np.array(changes_local_order_study_uni))
    changes_density_record_uni.append(np.array(changes_density_study_uni))
    
    changes_grobal_order_record_vicsek.append(np.array(changes_grobal_order_vicsek))
    changes_local_order_record_vicsek.append(np.array(changes_local_order_vicsek))
    changes_density_record_vicsek.append(np.array(changes_density_vicsek))
    
    progress_rate = (j+1)/11*100
    print(f"Progress: {progress_rate:.1f}%", end='\r')


#%%
# order - etaグラフ
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.ticker as mticker

CGOR = np.array(changes_grobal_order_record)
CLOR = np.array(changes_local_order_record)
CDR = np.array(changes_density_record)

CGOR_uni = np.array(changes_grobal_order_record_uni)
CLOR_uni = np.array(changes_local_order_record_uni)
CDR_uni = np.array(changes_density_record_uni)

CGOR_vicsek = np.array(changes_grobal_order_record_vicsek)
CLOR_vicsek = np.array(changes_local_order_record_vicsek)
CDR_vicsek = np.array(changes_density_record_vicsek)

CGOR = np.mean(CGOR[:, int(Tmax/2):], axis=1)
CLOR = np.mean(CLOR[:, int(Tmax/2):], axis=1)

CGOR_uni = np.mean(CGOR_uni[:, int(Tmax/2):], axis=1)
CLOR_uni = np.mean(CLOR_uni[:, int(Tmax/2):], axis=1)

CGOR_vicsek = np.mean(CGOR_vicsek[:, int(Tmax/2):], axis=1)
CLOR_vicsek = np.mean(CLOR_vicsek[:, int(Tmax/2):], axis=1)

fig6 = plt.figure(figsize=(40, 20))

ax = fig6.add_subplot(121)
ax.set_xlim(-0.1, 4.1)
ax.set_ylim(-0.1, 1.1)
ax.set_xticks(np.arange(0, 4.1, 0.4))
ax.set_yticks(np.arange(0, 1.1, 0.1))
ax.grid(True)
ax.set_title('eta - grobal order graph, number of agents={}'.format(num_agent), fontsize=14)
ax.set_xlabel('eta (= Noise intensity)', fontsize=14)
ax.set_ylabel('mean of grobal order in t > half of Tmax(={})'.format(Tmax), fontsize=14)
x = np.arange(0, 4.1, 0.4)
y = CGOR
y_uni = CGOR_uni
y_vicsek = CGOR_vicsek
ax.plot(x, y, '-or', label='Non Uniform Q-learning', alpha=0.5)
ax.plot(x, y_uni, '-ob', label='Uniform Q-learning', alpha=0.5)
ax.plot(x, y_vicsek, '-og', label='Vicsek modek', alpha=0.5)
ax.legend(fontsize=30)

ax2 = fig6.add_subplot(122)
ax2.set_xlim(-0.1, 4.1)
ax2.set_ylim(-0.1, 1.1)
ax2.set_xticks(np.arange(0, 4.1, 0.4))
ax2.set_yticks(np.arange(0, 1.1, 0.1))
ax2.grid(True)
ax2.set_title('eta - local order graph, number of agents={}'.format(num_agent), fontsize=14)
ax2.set_xlabel('eta (= Noise intensity)', fontsize=14)
ax2.set_ylabel('mean of local order in t > half of Tmax(={})'.format(Tmax), fontsize=14)
x = np.arange(0, 4.1, 0.4)
y = CLOR
y_uni = CLOR_uni
y_vicsek = CLOR_vicsek
ax2.plot(x, y, '-or', label='Non Uniform Q-learning', alpha=0.5)
ax2.plot(x, y_uni, '-ob', label='Uniform Q-learning', alpha=0.5)
ax2.plot(x, y_vicsek, '-og', label='Vicsek model', alpha=0.5)
ax2.legend(fontsize=30)

"""
ax3 = fig6.add_subplot(212)
ax3.set_xlim(0, Tmax)
#ax3.set_ylim(0, 1.0)
ax3.set_title('changes in density in each eta', fontsize=14)
ax3.set_xlabel('time', fontsize=14)
ax3.set_ylabel('density', fontsize=14)
colorbox = colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF', '#FF8000', '#0080FF', '#8000FF', '#FF0080', '#80FF00']

# カラーマップを作成
cmap = ListedColormap(colorbox)
# カラーバーを表示
sm = plt.cm.ScalarMappable(cmap=cmap)
sm.set_array([])  # カラーバー用のダミーの値を設定
cbar = plt.colorbar(sm, ax=ax3, orientation='vertical', pad=0.05, ticks=mticker.NullLocator()).set_label('The color of this color bar corresponds to the value of eta \n in 0.1 increments from the bottom, with red as 0 and lime as 1.', fontsize = 14)  # カラーバーを表示

for i in range(11):
    rate_CDR_i = CDR[i][1:] / CDR[i][:-1]
    r_gmean_i = np.exp(np.mean(np.log(rate_CDR_i)))
    ax3.plot(np.arange(Tmax), CDR[i], color=colorbox[i])
"""

plt.legend()
plt.show()

folder_name = "pmarl_2d_table_NonUniform_folder"
os.makedirs(folder_name, exist_ok=True)
#plt.savefig(os.path.join(folder_name, 'pmarl_2d_table_NonUniform_graphs_vs.png'), dpi=300)


#%%    
# QTable agent学習済みエージェントによるアニメーション作成用シミュレーション
def running_for_animation(agent):
    num_agent = 100
    perceptual_range = 1.0
    unit_length = np.round(np.sqrt(num_agent / 2.))
    lx_out = unit_length
    ly_out = unit_length
    v0 = 0.25
    eta = 0.0
    episodes = 800
    Tmax = 100
    action_size = 7
    state_size = 32
    policy = 'greedy'
    
    #data_R = np.load('sotsuken_MARL_nonuniform/data_R_sotsuken_MARL_nonuniform_sim7.npy')
    #data_v = np.load('sotsuken_MARL_nonuniform/data_v_sotsuken_MARL_nonuniform_sim7_5.npy')
    #print('data_R={}'.format(data_R))
    #data_eta = np.load('sotsuken_MARL_nonuniform/data_eta_sotsuken_MARL_nonuniform_sim7.npy')
    env = Enviroment_vicsek(action_size, num_agent, perceptual_range, lx_out, ly_out, v0, eta)
    
    positions = env.reset()

    x_history = []
    y_history = []
    angle_history = []

    changes_grobal_order_study = []
    changes_local_order_study = []
    changes_density_study = []

    action_record = []
    state_record = []

    for t in range(Tmax):
        x_vec, y_vec, angle_vec, _ = env.pos_to_vec(positions)
        x_history.append(x_vec)
        y_history.append(y_vec)
        angle_history.append(angle_vec)
            
        grobal_order, local_order, density = env.take_order_and_dencity(positions)
        changes_grobal_order_study.append(grobal_order)
        changes_local_order_study.append(np.mean(local_order))
        changes_density_study.append(np.mean(density))
            
        state = env.generate_state(positions)
        state_record.append(np.array(state))
            
        actions = agent.get_action(state, episodes)
        action_record.append(np.array(actions))
                
        next_positions, rewards, done = env.step(positions, actions)
        
        positions = next_positions
        
    return np.array(x_history), np.array(y_history), np.array(angle_history), np.array(changes_grobal_order_study), np.array(changes_local_order_study), np.array(changes_density_study)


#%%    
# vicsek modelエージェントによるアニメーション作成用シミュレーション
def running_for_vicsek_animation():
    num_agent = 100
    perceptual_range = 1.0
    unit_length = np.round(np.sqrt(num_agent / 2.))
    lx_out = unit_length
    ly_out = unit_length
    v0 = 0.5
    eta = 0.0
    Tmax = 500
    env_vicsek = Vicsek_model(num_agent, perceptual_range, v0, eta, lx_out, ly_out)

    x_history = []
    y_history = []
    angle_history = []

    changes_grobal_order_study = []
    changes_local_order_study = []
    changes_density_study = []
    
    x_vicsek, y_vicsek, theta_vicsek = env_vicsek.reset()

    for t in range(Tmax):
        #以下はvicsek modelのシミュレーションループ
        x_history.append(x_vicsek)
        y_history.append(y_vicsek)
        angle_history.append(theta_vicsek)
        
        grobal_order_vicsek, local_order_vicsek, density_vicsek = env_vicsek.take_parameter(x_vicsek, y_vicsek, theta_vicsek)
        next_x, next_y, next_theta = env_vicsek.step(x_vicsek, y_vicsek, theta_vicsek)
        
        x_vicsek, y_vicsek, theta_vicsek = next_x, next_y, next_theta
            
        changes_grobal_order_study.append(grobal_order_vicsek)
        changes_local_order_study.append(np.mean(local_order_vicsek))
        changes_density_study.append(np.mean(density_vicsek))  
        
    return np.array(x_history), np.array(y_history), np.array(angle_history), np.array(changes_grobal_order_study), np.array(changes_local_order_study), np.array(changes_density_study)


#%%
#学習過程における報酬総計の推移と１EPにおける各パラメータの推移を表示
import matplotlib.pyplot as plt

fig = plt.figure(figsize = (16, 40))

"""
ax = fig.add_subplot(511)
ax.grid(True)
ax.set_xlim(0, episodes)
ax.set_ylim(0, 100)
ax.set_title('Changes in Total Rewards', fontsize=14)
ax.set_xlabel('Episode', fontsize=14)
ax.set_ylabel('Total Reward', fontsize=14)

line, = ax.plot(np.arange(len(reward_history)), reward_history, color = 'r')
ax.legend([line], ['Reward'], loc='best', fontsize=18)


ax6 = fig.add_subplot(512)
ax6.grid(True)
ax6.set_xlim(0, episodes)
#ax6.set_ylim(0, 100)
ax6.set_title('Changes in num_collision', fontsize=14)
ax6.set_xlabel('Episode', fontsize=14)
ax6.set_ylabel('Number of collision', fontsize=14)

line6, = ax6.plot(np.arange(len(collision_mean_history)), collision_mean_history, color = 'r')
ax6.legend([line6], ['Number of collision'], loc='best', fontsize=18)
"""

ax2 = fig.add_subplot(513)
ax2.grid(True)
ax2.set_yticks(np.arange(0, 1.1, 0.1))
ax2.set_xlim(0, Tmax)
ax2.set_ylim(0, 1.0)
ax2.set_title('Changes in grobal order, mean of grobal order on t > half of Tmax={:.3f}'.format(np.mean(changes_grobal_order_study[int(Tmax/2):])), fontsize=14)
ax2.set_xlabel('Time', fontsize=14)
ax2.set_ylabel('grobal order', fontsize=14)

line_grobal_order, = ax2.plot(np.arange(len(changes_grobal_order_study)), changes_grobal_order_study, color = 'r')
ax2.legend([line_grobal_order], ['After learning'], loc='best', fontsize=18)

ax3 = fig.add_subplot(514)
ax3.grid(True)
ax3.set_yticks(np.arange(0, 1.1, 0.1))
ax3.set_xlim(0, Tmax)
ax3.set_ylim(0, 1.0)
ax3.set_title('Changes in local order, mean of local order in t > half of Tmax={:.3f}'.format(np.mean(changes_local_order_study[int(Tmax/2):])), fontsize=14)
ax3.set_xlabel('Time', fontsize=14)
ax3.set_ylabel('loacl order', fontsize=14)

line_local_order, = ax3.plot(np.arange(len(changes_local_order_study)), changes_local_order_study, color = 'r')
ax3.legend([line_local_order], ['After learning'], loc='best', fontsize=18)

ax4 = fig.add_subplot(515)
ax4.grid(True)
ax4.set_xlim(0, Tmax)
#ax4.set_ylim(0, 1.0)
ax4.set_title('Changes in density', fontsize=14)
ax4.set_xlabel('Time', fontsize=14)
ax4.set_ylabel('density', fontsize=14)

line_marl_density, = ax4.plot(np.arange(len(changes_density_study)), changes_density_study, color = 'r')
ax4.legend([line_marl_density], ['After learning'], loc='best', fontsize=18)

plt.show()



#%%
#アニメーションを作成
from matplotlib import animation
from IPython.display import HTML
import matplotlib.pyplot as plt

#file_name_data = np.loadtxt('pmarl_2d_table_NonUniform_folder/data_file_names_table_NonUniform_sim1.csv', delimiter=',', dtype='str')
num_agent = 100
num_clone = num_agent - 1
perceptual_range = 1.0
unit_length = np.round(np.sqrt(num_agent / 2.))
lx_out = unit_length
ly_out = unit_length
v0 = 0.25
eta = 0.0
episodes = 800
Tmax = 100
sync_interval = 20
action_size = 7
state_size = 32
policy = 'greedy'
agent = QTableAgent(state_size, action_size, num_agent, policy)

for j in range(1):
    # 1. モデルの保存ディレクトリとファイル名を指定する
    num = j
    #model_filename = file_name_data[num]
    #animation_tag = 'animation_table_sim9_' + str(num + 1) + '.gif'
    animation_tag = 'sotsuken_MARL_sim8.gif'

    # 2. モデルを読み込む
    agent.load_q_table('sotsuken_MARL_nonuniform/sotsuken_MARL_nonuniform_sim81.npy')
        
    # 4. シミュレーションを実行
    x, y, theta, cgos, clos, cds = running_for_animation(agent)
    #x, y, theta, cgos, clos, cds = running_for_vicsek_animation() # vicsek like modelのアニメーション

    #表示するエージェントのスケールを指定
    agent_scale = 0.01
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot()
    #data_R = np.load('sotsuken_MARL_nonuniform/data_R_sotsuken_MARL_nonuniform_sim7.npy')
    #use_R = data_R[0]
    #data_eta = np.load('sotsuken_MARL_nonuniform/data_eta_sotsuken_MARL_nonuniform_sim7.npy')
    #use_eta = data_eta[0]
    #data_v = np.load('sotsuken_MARL_nonuniform/data_v_sotsuken_MARL_nonuniform_sim7_5.npy')
    use_v = data_v[0]
    
    # カラーバーを作成
    #cbar = plt.colorbar(ax1.scatter([], [], c=[], cmap='viridis'))
    #cbar.set_label('v Value', fontsize=12)

    #アップデート関数
    def update1(i):
        data_x1 = x[i][:]
        data_y1 = y[i][:]
        data_angle1 = theta[i][:]
        data_grobal = cgos[i]
        #data_local = clos[i]
        #data_density = cds[i]
        
        ax1.clear()
        
        ax1.set_xlim(0, lx_out)
        ax1.set_ylim(0, ly_out)
        
        #各タイトル
        #ax1.set_title('N={}, R={}, episodes={}, Tmax={}, v={}, eta={}, t={}'.format(num_agent, perceptual_range, episodes, Tmax, v0, eta, i+1) + '\nGrobal order={:.3f}, Local order={:.3f}, Density={:.3f}'.format(data_grobal, data_local, data_density), fontsize=14)
        ax1.set_title('N={}, Tmax={}, v={}, t={}'.format(num_agent, Tmax, v0, i+1) + ', Order parameter={:.3f}'.format(data_grobal), fontsize=14)

        #エージェントベクトル更新
        #scatter = ax1.scatter(data_x1, data_y1, c=use_v, cmap='viridis', s=100)
        ax1.quiver(data_x1, data_y1, agent_scale * np.cos(data_angle1), agent_scale * np.sin(data_angle1), scale=0.5)
        
        # カラーバーの更新
        #cbar.update_normal(scatter)
    
        # 進捗状況を出力する
        progress_rate = (i+1)/Tmax*100
        print("{}th, Animation progress={:.3f}%".format(j+1, progress_rate), end='\r')
        
    #アニメーション作成とgif保存
    ani = animation.FuncAnimation(fig, update1, frames=range(Tmax))

    #グラフ表示
    plt.show()

    #アニメーションの保存
    output_file = os.path.join('sotsuken_MARL_nonuniform', animation_tag)
    ani.save(output_file, writer='pillow')

    #アニメーションを表示
    #HTML(ani.to_jshtml())


#%%
#各ヒートマップを一括表示する
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#各種パラメータ
file_name_data = np.loadtxt('pmarl_2d_table_NonUniform_folder/data_file_names_table_NonUniform_sim1_2.csv', delimiter=',', dtype='str')
data_R = np.load('pmarl_2d_table_NonUniform_folder/data_R_table_NonUniform_sim1_2.npy')
num_agent = 100
num_clone = num_agent - 1
perceptual_range = 1.0
unit_length = np.round(np.sqrt(num_agent / 2.))
lx_out = unit_length
ly_out = unit_length
v0 = 0.5
eta = 0.0
episodes = 800
Tmax = 10000
sync_interval = 20
action_size = 7
state_size = 32
policy = 'greedy'

#agent = QTableAgent(state_size, action_size, num_agent, policy)
#env = Enviroment_vicsek(action_size, num_agent, perceptual_range, lx_out, ly_out, v0, eta)

#index = file_name_data[0]
#QTable = np.load('pmarl_2d_table_NonUniform_folder/' + index)

R_index = data_R[0]

# キャンバスのサイズとレイアウトを設定
num_rows = 4  # 行数
num_cols = 4  # 列数
fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 25))

for k in range(num_rows * num_cols):
    # 検査対象の個体群を選択(1 ~ 20で選択)
    num = 4*k
    index = file_name_data[int(k / 4)]
    QTable = np.load('pmarl_2d_table_NonUniform_folder/' + index)
    Q = QTable[num].T

    # ヒートマップを作成
    row = k // num_cols  # 行
    col = k % num_cols  # 列
    ax = axes[row, col]
    im = ax.imshow(Q, cmap='jet', origin='lower', aspect='auto')
    #ax.set_xticks([-np.pi, -2/3*np.pi, -1/3*np.pi, 0, 1/3*np.pi, 2/3*np.pi, np.pi])
    #ax.set_xticklabels(['-π', '-2π/3','-π/3' '0', 'π'])
    #ax.set_yticks(np.linspace(0, 3, 7))
    #ax.set_yticklabels([r'$-\theta_{max}$', '', '', '0', '', '', r'$\theta_{max}$'])
    ax.set_title("sim={}, Agent {}, R={}".format(int(k / 4), num, R_index[num]))
    plt.colorbar(im, ax=ax)

# タイトルが重複しないように調整
#plt.suptitle("Heatmaps of Q function values for 5 Agents.\nHowever, the value of the Q function at each state is normalized by the following equation.\n" + r"$Q_{normalized} = \frac{Q - Q_{min}}{Q_{max} - Q_{min}}$", fontsize=20)
#plt.suptitle("Heatmaps of Q function values for 5 Agents.\nHowever, the maximum value is set to 1 and other values are scaled to 0.", fontsize=20)
plt.suptitle("Heatmaps of Q Table", fontsize=20)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# キャンバスを保存
folder_name = "pmarl_2d_table_NonUniform_folder"
os.makedirs(folder_name, exist_ok=True)
plt.savefig(os.path.join(folder_name, "QTable_some_agents_table_NonUniform_sim5.png"), dpi=300)


#%%
#各ヒートマップを一括表示する
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#各種パラメータ
file_name_data = np.loadtxt('sotsuken_MARL_nonuniform/data_file_names_sotsuken_MARL_nonuniform_sim2.csv', delimiter=',', dtype='str')
#data_R = np.load('sotsuken_MARL_nonuniform/data_R_sotsuken_MARL_nonuniform_sim2.npy')
num_agent = 100
num_clone = num_agent - 1
perceptual_range = 1.0
unit_length = np.round(np.sqrt(num_agent / 2.))
lx_out = unit_length
ly_out = unit_length
v0 = 0.5
eta = 0.0
episodes = 800
Tmax = 10000
sync_interval = 20
action_size = 7
state_size = 32
policy = 'greedy'

#agent = QTableAgent(state_size, action_size, num_agent, policy)
#env = Enviroment_vicsek(action_size, num_agent, perceptual_range, lx_out, ly_out, v0, eta)

#index = file_name_data[0]
#QTable = np.load('pmarl_2d_table_NonUniform_folder/' + index)

#R_index = data_R[0]

# キャンバスのサイズとレイアウトを設定
num_rows = 5  # 行数
num_cols = 4  # 列数
fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 25))

for k in range(num_rows * num_cols):
    # 検査対象の個体群を選択(1 ~ 20で選択)
    num = 5*k
    index = file_name_data[int(k / 9)]
    QTable = np.load('sotsuken_MARL_nonuniform/' + index)
    Q = QTable[num].T
    
    index = np.argmin(Q, axis=0)

    # ヒートマップを作成
    row = k // num_cols  # 行
    col = k % num_cols  # 列
    ax = axes[row, col]
    im = ax.imshow(Q, cmap='jet', origin='lower', aspect='auto')
    ax.set_xticks([0, 15, 30])
    ax.set_xticklabels(['-π', '0', 'π'])
    ax.set_yticks([0,1,2,3,4,5,6])
    ax.set_yticklabels([r'$-\theta_{max}$', '', '', '0', '', '', r'$\theta_{max}$'])
    #ax.set_title("R={:.3f}".format(R_index[num]))
    ax.set_title("R={:.3f}".format(1.0))
    ax.plot(np.arange(Q.shape[1]-1), index[:-1], '-x', color='white')
    plt.colorbar(im, ax=ax)

# タイトルが重複しないように調整
#plt.suptitle("Heatmaps of Q function values for 5 Agents.\nHowever, the value of the Q function at each state is normalized by the following equation.\n" + r"$Q_{normalized} = \frac{Q - Q_{min}}{Q_{max} - Q_{min}}$", fontsize=20)
#plt.suptitle("Heatmaps of Q function values for 5 Agents.\nHowever, the maximum value is set to 1 and other values are scaled to 0.", fontsize=20)
plt.suptitle("Heatmaps of Q Table", fontsize=20)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# キャンバスを保存
folder_name = "sotsuken_MARL_nonuniform"
os.makedirs(folder_name, exist_ok=True)
plt.savefig(os.path.join(folder_name, "QTable_sotsuken_MARL_sim2.png"), dpi=300)




#%%
#ヒートマップに表示
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sample_index = np.random.choice(Tmax, int(Tmax / 2), replace=False)
sample_state = state_record[sample_index].flatten()
sample_action = action_record[sample_index].flatten()

plt.rcParams["font.size"] = 15
fig = plt.figure()
ax = fig.add_subplot(111)

H = ax.hist2d(sample_state, sample_action, bins=[40, 3], cmap=cm.jet)
ax.set_title('{}th Agent'.format(num), fontsize=20)
ax.set_xlabel('State', fontsize=20)
ax.set_xticks([-np.pi, 0, np.pi])
ax.set_xticklabels(['-π', '0', 'π'])
 
ax.set_ylabel('Action', fontsize=20)
ax.set_yticks(np.linspace(0, 3, 7))
ax.set_yticklabels(['', r'$-\theta_{max}$', '', '0', '', r'$\theta_{max}$', ''])
cbar = plt.colorbar(H[3],ax=ax)
cbar.set_label('Frequency', fontsize=20)
plt.show()


#%%
#actionとstateの頻度を棒グラフとヒストグラムでアニメーションを作成
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML

plt.rcParams["font.size"] = 15
fig = plt.figure(figsize=(10, 20))
ax = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

bin = np.round(1 + np.log2(num_agent))

def update(i):
    ax.clear()
    ax2.clear()
    
    
    sample_state = state_record[i]
    sample_action = action_record[i]
    data_grobal = grobal_record[i]
    data_local = local_record[i]
    data_density = density_record[i]
    
    data_0 = np.where(sample_action == 0, 1, 0).sum()
    data_1 = np.where(sample_action == 1, 1, 0).sum()
    data_2 = np.where(sample_action == 2, 1, 0).sum()
    data_count = np.array([data_0, data_1, data_2])
    
    ax.set_title('{}th Agent, t={}\nGrobal order={:.3f}, Local order={:.3f}, Density={:.3f}'.format(num, i + 1, data_grobal, data_local, data_density), fontsize=20)
    x_labels = [r'$-\theta_{max}$', '0', r'$\theta_{max}$']
    ax.bar(np.arange(3), data_count, tick_label=x_labels, align='center')
    ax.set_xlabel('Action', fontsize=20)
    ax.set_ylim(0, num_agent)
    ax.set_ylabel('Number of Agents', fontsize=20)
    
    ax2.set_title('{}th Agent, t={}\nGrobal order={:.3f}, Local order={:.3f}, Density={:.3f}'.format(num, i + 1, data_grobal, data_local, data_density), fontsize=20)
    ax2.hist(sample_state, bins=np.linspace(-np.pi, np.pi, int(bin)))
    ax2.set_xlabel('State', fontsize=20)
    #ax2.set_xticks(np.arange(-np.pi, np.pi, 2*bin + 1))
    #ax2.set_xticklabels(['-π', '', '', '', '0', '', '', '', 'π'])
    ax2.set_xlabel('State', fontsize=20)
    ax2.set_xlim(-np.pi, np.pi)
    ax2.set_ylim(0, num_agent)
    ax2.set_ylabel('Number of Agents', fontsize=20)
    
    
    print('Progress={:.3}%'.format((i+1)/Tmax * 100), end='\r')
    
ani = animation.FuncAnimation(fig, update, frames=range(Tmax))
HTML(ani.to_jshtml())


#%% 
#エージェントの遭遇するstateの分布を算出する
#学習済みエージェントによるシミュレーションをnum_test回繰り返して平均を取る
import numpy as np
import torch

#実行関数の定義
def running_simulation_tmax(num, num_test, agent, Tmax, num_agent):
    reward_record_mean = []
    local_order_record_mean = []
    grobal_order_record_mean = []
    density_record_mean = []
    
    state_history = []
    action_history = []
    
    for j in range(num_test):
        agents = [agent]
        for i in range(num_agent - 1):
            clone_i = copy.deepcopy(agent)
            agents.append(clone_i)
        positions = env.reset()
    
        reward_records = []
        local_order_record = []
        grobal_order_record = []
        density_record = []
        
        state_record = []
        action_record = []
    
        for t in range(Tmax):
            state = env.generate_state(positions)
            state_record.append(state.reshape((num_agent,)))
            
            grobal_order, local_orders, density = env.take_order_and_dencity(positions)
            grobal_order_record.append(grobal_order)
            local_order_record.append(np.mean(local_orders))
            density_record.append(np.mean(density))
            
            actions = []
            for i in range(num_agent):
                agent_i = agents[i]
                state_i = state[i]
                action_i = agent_i.get_action(state_i)
                actions.append(action_i)
            action_record.append(np.array(actions))
                    
            next_positions, rewards, done = env.step(positions, actions)
            reward_records.append(np.mean(rewards))
        
            positions = next_positions
        
        reward_record_mean.append(np.array(reward_records))
        local_order_record_mean.append(np.array(local_order_record))
        grobal_order_record_mean.append(np.array(grobal_order_record))
        density_record_mean.append(np.array(density_record))
        
        state_history.append(np.array(state_record)) # (300 x Tmax) x num_test
        action_history.append(np.array(action_record)) # (300 x Tmax) x num_test
    
        progress_rate = (j+1)/num_test*100
        print('{}th Progress={:.1f}%'.format(num + 1, progress_rate), end='\r')
        
    return reward_record_mean, local_order_record_mean, grobal_order_record_mean, density_record_mean, np.array(state_history), np.array(action_history)



#%%
#エージェントの遭遇するstateについての分布を算出する
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#各種パラメータ
num_agent = 300
num_clone = num_agent - 1
perceptual_range = 2.5
lx_out = 30
ly_out = 30
v0 = 1.0
eta = 0.0
episodes = 100
Tmax_sample = 100
sync_interval = 20
action_size = 3
state_size = 1
policy = 'epsilon_greedy'
env = Enviroment_vicsek(num_agent, perceptual_range, lx_out, ly_out, v0, eta)
num_test = 20

#エージェントを召喚
agent = dqn_agent(action_size, state_size, policy)

#シミュレーションを実行
data_set_state = []
data_set_action = []
for i in range(11):
    Tmax = Tmax_sample + i * 100
    _, _, _, _, state_history, action_history = running_simulation_tmax(i, num_test, agent, Tmax, num_agent)
    data_set_state.append(state_history)
    data_set_action.append(action_history)


#%%
#ヒートマップを作成
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#フォルダの作成
folder_name = "pmarl_2d_new_folder"
os.makedirs(folder_name, exist_ok=True)

for i in range(11):
    # サブプロットを作成
    # キャンバスのサイズとレイアウトを設定
    num_rows = 6  # 行数
    num_cols = 10  # 列数
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(40, 24))
    numbers = np.random.choice(num_agent, int(num_rows * (num_cols / 2)), replace=False)
    state_record = data_set_state[i]
    Tmax = i * 100
    tag = 'distribution_state_Tmax_' + str() + '.png'

    for k in range(int(num_rows * (num_cols / 2))):
        # i番目のエージェントに対して作成するためのデータを抽出
        num = numbers[k]
        num_row = 2 * k
        num_col = 2 * k
        state_history_agent1 = state_record[:, :, num]  # (Tmax x num_test)

        # ヒートマップをプロット
        row = num_row // num_cols  # 行
        col = num_col % num_cols  # 列
        ax1 = axes[row, col]
        ax2 = axes[row, col + 1]
        num_test_vec = []
        for i in range(num_test):
            row_i = np.full(Tmax, i + 1)
            num_test_vec.append(row_i)
        num_test_vec = np.array(num_test_vec).flatten()
        H1 = ax1.hist2d(state_history_agent1.flatten(), num_test_vec, bins=[100, 20], cmap='magma')
        #ax1.set_ylabel('Trial number (bins=20)')
        ax1.set_yticks([1, 10, 20])
        ax1.set_yticklabels(['1', '10', '20'])
        ax1.set_xticks([-np.pi, 0, np.pi])
        ax1.set_xticklabels(['-π', '0', 'π'])
        #ax1.set_xlabel('State (bins=100)')
        cbar = plt.colorbar(H[3], ax=ax1)
        cbar.set_label('Frequency', fontsize=12)
        #ax1.set_title("Distribution of states encountered by unlearned agents.\nTime steps={}, Number of Agents={}".format(Tmax, num_agent), fontsize=14)

        # カーネル密度推定をプロット
        sns.kdeplot(data=state_history_agent1.flatten(), ax=ax2)
        #ax2.set_xlabel('State')
        ax2.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        ax2.set_xticklabels(['-π', r'$-\frac{\pi}{2}$', '0', r'$\frac{\pi}{2}$', 'π'])
        #ax2.set_title('Distribution of probability of encountering state\nby kernel density estimation', fontsize=14)

    plt.suptitle("Distribution of states encountered by unlearned agents And Distribution of probability of encountering state by kernel density estimation.\n" + 'Heat map : Y axis is trial number, X axis is state, Grahp : Y axis is probability density, X axis is state\n' + r'System size=$[0, {}]^2$, Time steps={}, Number of Agents={}, v={}, R={}, eta={}'.format(lx_out, Tmax, num_agent, v0, perceptual_range, eta), fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(folder_name, tag), dpi=300)

# プロットを表示
#plt.tight_layout()
#plt.show()

#%%
print(data_set_state[1].shape)
print(data_set_state[1][:, :, 1].shape)
#%%
"""
#ヒートマップを作成
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#1番目のエージェントに対して作成するためのデータを抽出
state_history_agent1 = state_history[:, :, 0] #(Tmax x num_test)

#hist2dを使用
num_test_vec = []
for i in range(num_test):
    row_i = np.full(Tmax, i + 1)
    num_test_vec.append(row_i)
num_test_vec = np.array(num_test_vec).flatten()
H = plt.hist2d(state_history_agent1.flatten(), num_test_vec, bins=[100, 20], cmap='magma')
plt.ylabel('Trial number (bins=20)')
plt.yticks([1, 10, 20], ['1', '10', '20'])
plt.xticks([-np.pi, 0, np.pi], ['-π', '0', 'π'])  # xticklabels の修正
plt.xlabel('State (bins=100)')
cbar = plt.colorbar(H[3])
cbar.set_label('Frequency', fontsize=20)
plt.title("Distribution of states encountered by unlearned agents.\nTime steps={}, Number of Agents={}".format(Tmax, num_agent), fontsize=14)
plt.show()

#カーネル密度推定
import seaborn as sns
sns.kdeplot(data=state_history_agent1.flatten())
plt.xlabel('State')
plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], ['-π', r'$-\frac{\pi}{2}$', '0', r'$\frac{\pi}{2}$', 'π'])  # xticklabels の修正
plt.title('Distribution of probability of encountering state\nby kernel density estimation')
"""

# %%
#vicsek modelによるシミュレーションをnum_test回繰り返して平均を取る
num_agent = 300
num_clone = num_agent - 1
perceptual_range = 2.5
lx_out = 30
ly_out = 30
v0 = 1.0
eta = 0.0
num_test = 20
Tmax = 200

env_vicsek = Vicsek_model(num_agent, perceptual_range, v0, eta, lx_out, ly_out)

local_order_record_mean = []
density_record_mean = []
grobal_order_record_mean = []

for j in range(num_test):
    x_vicsek, y_vicsek, theta_vicsek = env_vicsek.reset()
    
    local_order_record = []
    density_record = []
    grobal_order_record = []
    
    for t in range(Tmax):
        grobal_order_vicsek, local_order_vicsek, density_vicsek = env_vicsek.take_parameter(x_vicsek, y_vicsek, theta_vicsek)
        next_x, next_y, next_theta = env_vicsek.step(x_vicsek, y_vicsek, theta_vicsek)
        
        x_vicsek, y_vicsek, theta_vicsek = next_x, next_y, next_theta
        
        grobal_order_record.append(grobal_order_vicsek)
        local_order_record.append(local_order_vicsek)
        density_record.append(density_vicsek)
    
    local_order_record_mean.append(np.array(local_order_record))
    grobal_order_record_mean.append(np.array(grobal_order_record))
    density_record_mean.append(np.array(density_record))
    
    progress_rate = (j+1)/num_test*100
    print(f"Progress: {progress_rate:.1f}%", end='\r')
    

# %%
#学習済みエージェントによる20回のシミュレーションにおいて得られた報酬の推移の平均を描画
import matplotlib.pyplot as plt

fig3 = plt.figure(figsize = (16,32))

LORM = np.array(local_order_record_mean)
LORM_mean = np.mean(LORM, axis=0)

GORM = np.array(grobal_order_record_mean)
GORM_mean = np.mean(GORM, axis=0)

DRM = np.array(density_record_mean)
DRM_mean = np.mean(DRM, axis=0)

ax2 = fig3.add_subplot(311)
ax2.grid(True)
ax2.set_yticks(np.arange(0, 1.1, step=0.1))
ax2.set_xlim(0, Tmax)
ax2.set_ylim(0, 1)
ax2.set_title('Changes in mean of local order, N={}, Tmax={}, Number of trial={}, mean in t > 100={:.3f}'.format(num_agent, Tmax, num_test, np.mean(LORM_mean[100:])), fontsize=14)
ax2.set_xlabel('time', fontsize=14)
ax2.set_ylabel('local order \n local order is an order parameter within the local radius of each agent.', fontsize=14)

line2, = ax2.plot(np.arange(len(LORM_mean)), LORM_mean, color = 'k')
for i in range(num_test):
    ax2.plot(np.arange(len(LORM[i])), LORM[i], color='green', alpha=0.1)
ax2.legend([line2], ['mean of local order'], loc='lower right', fontsize=18)

ax4 = fig3.add_subplot(312)
ax4.grid(True)
ax4.set_yticks(np.arange(0, 1.1, step=0.1))
ax4.set_xlim(0, Tmax)
ax4.set_ylim(0, 1)
ax4.set_title('Changes in mean of grobal order, N={}, Tmax={}, Number of trial={}, mean in t > 100={:.3f}'.format(num_agent, Tmax, num_test, np.mean(GORM_mean[100:])), fontsize=14)
ax4.set_xlabel('time', fontsize=14)
ax4.set_ylabel('grobal order \n grobal order is an order parameter for the entire system.', fontsize=14)

line4, = ax4.plot(np.arange(len(GORM_mean)), GORM_mean, color = 'k')
for i in range(num_test):
    ax4.plot(np.arange(len(GORM[i])), GORM[i], color='green', alpha=0.1)
ax4.legend([line2], ['mean of grobal order'], loc='lower right', fontsize=18)

ax3 = fig3.add_subplot(313)
ax3.grid(True)
ax3.set_xlim(0, Tmax)
#ax3.set_ylim(0, 1.0)
ax3.set_title('Changes in mean of density, N={}, Tmax={}, Number of trial={}'.format(num_agent, Tmax, num_test), fontsize=14)
ax3.set_xlabel('time', fontsize=14)
ax3.set_ylabel('density \n Density is a parameter that indicates the density of agents within a local radius. ', fontsize=14)

line3, = ax3.plot(np.arange(len(DRM_mean)), DRM_mean, color = 'k')
for i in range(num_test):
    ax3.plot(np.arange(len(DRM[i])), DRM[i], color='green', alpha=0.1)
ax3.legend([line3], ['mean of dencity'], loc='best', fontsize=18)

plt.show()

# %%
