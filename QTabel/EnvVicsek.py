
"""
目標：
卒研で用いたエージェントの学習環境の再構築。学習プロトコルの実装。

参考資料：
[1] 卒研2024

"""

#%%
# 学習環境の設定
import numpy as np

class EnvironmentVicsek:
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
    
    def pBC(self, x, l):
        return np.where(x < 0, x + l, np.where(x > l, x - l, np.where((-l <= x) & (x <= l), x, 0)))
    
    def pos_to_vec(self, pos):
        return pos.T

    def see_forward(self, position):
        x, y, theta, _ = self.pos_to_vec(position)
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dist = np.sqrt(dx**2 + dy**2)
        index_neighbor = self.cul_neighbor(position)
        
        dot = (dx * np.cos(theta).reshape((self.num_agent, 1)) + dy * np.sin(theta).reshape((self.num_agent, 1))) / dist
        index_insight = np.arccos(dot) < (3/16) * np.pi
        
        return index_neighbor * index_insight
    
    def num_to_theta(self, action):
        action_list = np.linspace(start=-(3/16)*np.pi, stop=(3/16)*np.pi, num=self.action_size)
        return action_list[action]
    
    def get_s_next(self, s, a):
        x, y, theta, R = self.pos_to_vec(s)
        theta_add = self.num_to_theta(a)
        
        theta_next = theta + theta_add + np.random.uniform(low=-(1/2), high=(1/2), size=self.num_agent) * self.eta
        theta_next = np.arctan2(np.sin(theta_next), np.cos(theta_next))
        x_next = x + self.v0 * np.cos(theta_next)
        y_next = y + self.v0 * np.sin(theta_next)
        
        x_next = self.pBC(x_next, self.lx_out)
        y_next = self.pBC(y_next, self.ly_out)
        
        return np.array([x_next, y_next, theta_next, R]).T
    
    def cul_neighbor(self, s):
        x, y, _, R = self.pos_to_vec(s)
        R_m = np.tile(R, (self.num_agent, 1)).T
        dx = np.abs(x[np.newaxis, :] - x[:, np.newaxis])
        dx = np.where(dx > self.lx_out / 2, dx - self.lx_out, dx)
        dy = np.abs(y[np.newaxis, :] - y[:, np.newaxis])
        dy = np.where(dy > self.ly_out / 2, dy - self.ly_out, dy)
        return np.sqrt(dx**2 + dy**2) < R_m
    
    def take_order_and_density(self, s):
        neighbor_matrix = self.cul_neighbor(s)
        _, _, theta, R = self.pos_to_vec(s)
        cos = np.cos(theta)
        sin = np.sin(theta)
        
        cos_neighbor = np.dot(neighbor_matrix, cos)
        sin_neighbor = np.dot(neighbor_matrix, sin)
        num_neighbors = neighbor_matrix.sum(axis=1)
        
        order_norm = np.sqrt(cos_neighbor**2 + sin_neighbor**2)
        local_order = np.where(num_neighbors != 0, order_norm / num_neighbors, 0)
        
        density = num_neighbors / (R**2 * np.pi)
        global_order = (1 / self.num_agent) * np.sqrt(np.sum(cos)**2 + np.sum(sin)**2)
        
        return global_order, local_order, density
    
    def generate_position(self):
        x = np.random.uniform(low=0, high=self.lx_out, size=self.num_agent)
        y = np.random.uniform(low=0, high=self.ly_out, size=self.num_agent)
        theta = np.random.uniform(low=-np.pi, high=np.pi, size=self.num_agent)
        R = np.full(self.num_agent, self.radius)
        return np.array([x, y, theta, R]).T
    
    def reset(self):
        return self.generate_position()
    
    def flag_coll(self, s):
        x, y, _, _ = self.pos_to_vec(s)
        dx = x[np.newaxis, :] - x[:, np.newaxis]
        dy = y[np.newaxis, :] - y[:, np.newaxis]
        dist = np.sqrt(dx**2 + dy**2)
        return dist < 0.3
    
    def step(self, s, a):
        past_position = np.array(s)
        next_position = self.get_s_next(s, a)
        
        neighbor_matrix_past = self.cul_neighbor(past_position).sum(axis=1)
        neighbor_matrix_next = self.cul_neighbor(next_position).sum(axis=1)
        reward = np.where(neighbor_matrix_past > neighbor_matrix_next, 1, 0)
        
        return np.array(next_position), reward, False
    
    def arg(self, s):
        x, y, theta, _ = self.pos_to_vec(s)
        dis_cul = self.cul_neighbor(s)
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
        
        theta_r = np.pi/2
        rota = np.array([[np.cos(theta_r), -np.sin(theta_r)], [np.sin(theta_r), np.cos(theta_r)]])
        rota_v = np.tile(rota, (self.num_agent, 1, 1))
        vTop = v.reshape((self.num_agent, 2, 1))
        rota_dot_vTop = np.matmul(rota_v, vTop).reshape((self.num_agent, 2))
        P_dot_vTop = (P * rota_dot_vTop).sum(axis=1)
        
        naiseki = (P_normalize * v).sum(axis=1)
        arccos = np.arccos(naiseki)
        return np.where(P_dot_vTop > 0, arccos, -arccos)
    
    def generate_state(self, position):
        return self.arg(position)
