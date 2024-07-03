#%%
# boid modelの実装
"""
 ▄▄       ▄▄  ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄   ▄▄▄▄▄▄▄▄▄▄▄  ▄           
▐░░▌     ▐░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░▌ ▐░░░░░░░░░░░▌▐░▌          
▐░▌░▌   ▐░▐░▌▐░█▀▀▀▀▀▀▀█░▌▐░█▀▀▀▀▀▀▀█░▌▐░█▀▀▀▀▀▀▀▀▀ ▐░▌          
▐░▌▐░▌ ▐░▌▐░▌▐░▌       ▐░▌▐░▌       ▐░▌▐░▌          ▐░▌          
▐░▌ ▐░▐░▌ ▐░▌▐░▌       ▐░▌▐░▌       ▐░▌▐░█▄▄▄▄▄▄▄▄▄ ▐░▌          
▐░▌  ▐░▌  ▐░▌▐░▌       ▐░▌▐░▌       ▐░▌▐░░░░░░░░░░░▌▐░▌          
▐░▌   ▀   ▐░▌▐░▌       ▐░▌▐░▌       ▐░▌▐░█▀▀▀▀▀▀▀▀▀ ▐░▌          
▐░▌       ▐░▌▐░▌       ▐░▌▐░▌       ▐░▌▐░▌          ▐░▌          
▐░▌       ▐░▌▐░█▄▄▄▄▄▄▄█░▌▐░█▄▄▄▄▄▄▄█░▌▐░█▄▄▄▄▄▄▄▄▄ ▐░█▄▄▄▄▄▄▄▄▄ 
▐░▌       ▐░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░▌ ▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌
 ▀         ▀  ▀▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀   ▀▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀▀ 
"""
import numpy as np
import random

class Boid:
    # 諸定数
    def __init__(self, num, r_s, r_a, r_c, w_s, w_a, w_c, lx, ly):
        self.num = num
        self.r_s = r_s
        self.r_a = r_a
        self.r_c = r_c
        self.w_s = w_s
        self.w_a = w_a
        self.w_c = w_c
        self.lx = lx
        self.ly = ly

        # 時間刻みはば
        self.dt = 0.1
        # 固定最大速度
        self.v_max = 3.0

    # 周期境界条件
    def pbc(self, x, y):
        output_x = np.where(x < 0, x + self.lx, np.where(self.lx < x, x - self.lx, x))
        output_y = np.where(y < 0, y + self.ly, np.where(self.ly < y, y - self.ly, y))
        return output_x, output_y
    
    # 周期境界条件下での距離行列の計算
    def pbc_ad(self, dx, dy, range):
        #dx_sing = dx > 0
        #dy_sing = dy > 0
        dx_abs = np.abs(dx)
        dy_abs = np.abs(dy)
        dx_abs_inv = np.abs(self.lx - dx_abs)
        dy_abs_inv = np.abs(self.ly - dy_abs)

        index_x = dx_abs > dx_abs_inv
        index_y = dy_abs > dy_abs_inv
        #dx_abs_true = np.where(index_x == True, dx_abs_inv, dx_abs)
        #dy_abs_true = np.where(index_y == True, dy_abs_inv, dy_abs)
        dx = np.where(index_x == True, -dx_abs_inv, dx)
        dy = np.where(index_y == True, -dy_abs_inv, dy)
        dist = np.sqrt(dx**2 + dy**2)
        ad_matrix = dist < range
        return dx, dy, dist, ad_matrix

    # 初期位置と初期速度の生成
    def reset(self):
        x = np.random.uniform(high=self.lx, low=0, size=self.num)
        y = np.random.uniform(high=self.lx, low=0, size=self.num)
        vx = np.random.uniform(size=self.num) * 0.1
        vy = np.random.uniform(size=self.num) * 0.1
        return x, y, vx, vy
    
    # 正規化計算式
    def normalize(self, dx, dy):
        norm = np.sqrt(dx**2 + dy**2)
        dx_normal = np.where(norm != 0, dx / norm, 0)
        dy_normal = np.where(norm != 0, dy / norm, 0)
        return dx_normal, dy_normal
    
    # 隣接行列らの計算
    def neighbor(self, x, y, range):
        dx = x[np.newaxis,:] - x[:,np.newaxis] # x_i - x_j, i is me
        dy = y[np.newaxis,:] - y[:,np.newaxis]
        #dist = self.pbc_dist(np.sqrt(dx**2 + dy**2))
        #neighbor_index = dist < range
        #ad_matrix = neighbor_index - np.eye(self.num)

        dx, dy, dist, ad_matrix = self.pbc_ad(dx, dy, range)
        return dx, dy, dist, ad_matrix
    
    # 分離の計算
    def separate(self, x, y):
        dx, dy, dist, ad_matrix = self.neighbor(x, y, self.r_s)
        dist_neighbor = np.sum(dist * ad_matrix, axis=1)
        dx_neighbor = np.sum(dx * ad_matrix, axis=1)
        dy_neighbor = np.sum(dy * ad_matrix, axis=1)

        fx = np.where(dist_neighbor != 0, dx_neighbor / dist_neighbor, 0)
        fy = np.where(dist_neighbor != 0, dy_neighbor / dist_neighbor, 0)
        return -fx, -fy
    
    # 整列の計算
    def alignment(self, x, y, vx, vy):
        dx, dy, dist, ad_matrix = self.neighbor(x, y, self.r_a)
        num_neighbor = np.sum(ad_matrix, axis=1)
        vx_neighbor = vx[:,np.newaxis] * ad_matrix
        vy_neighbor = vy[:,np.newaxis] * ad_matrix

        vx_a = np.where(num_neighbor != 0, np.sum(vx_neighbor, axis=1) / num_neighbor, 0)
        vy_a = np.where(num_neighbor != 0, np.sum(vy_neighbor, axis=1) / num_neighbor, 0)
        fx = vx_a - vx
        fy = vy_a - vy
        return fx, fy
    
    # 結合の計算
    def cohesion(self, x, y):
        dx, dy, dist, ad_matrix = self.neighbor(x, y, self.r_c)
        num_neighbor = np.sum(ad_matrix, axis=1)
        dx_neighbor = np.sum(dx * ad_matrix, axis=1)
        dy_neighbor = np.sum(dy * ad_matrix, axis=1)

        fx = np.where(num_neighbor != 0, dx_neighbor / num_neighbor, 0)
        fy = np.where(num_neighbor != 0, dy_neighbor / num_neighbor, 0)
        return fx, fy
    
    # 時間を進める
    def step(self, x, y, vx, vy):
        fx_s, fy_s = self.separate(x, y)
        fx_a, fy_a = self.alignment(x, y, vx, vy)
        fx_c, fy_c = self.cohesion(x, y)

        # 正規化
        fx_s_normal, fy_s_normal = self.normalize(fx_s, fy_s)
        fx_a_normal, fy_a_normal = self.normalize(fx_a, fy_a)
        fx_c_normal, fy_c_normal = self.normalize(fx_c, fy_c)
        
        # f = ma
        fx = self.w_s * fx_s_normal + self.w_a * fx_a_normal + self.w_c * fx_c_normal
        fy = self.w_s * fy_s_normal + self.w_a * fy_a_normal + self.w_c * fy_c_normal

        # 速度の更新：v_t+1 = v_t + f * dt
        vx_next = vx + fx * self.dt
        vy_next = vy + fy * self.dt
        v_next = np.sqrt(vx_next**2 + vy_next**2)
        vx_next = np.where(v_next > self.v_max, vx_next / v_next * self.v_max, vx_next)
        vy_next = np.where(v_next > self.v_max, vy_next / v_next * self.v_max, vy_next)

        # 位置の更新：p_t+1 = p_t + v_t+1 * dt
        x_next = x + vx_next * self.dt
        y_next = y + vy_next * self.dt
        x_next_pbc, y_next_pbc = self.pbc(x_next, y_next)
        return x_next_pbc, y_next_pbc, vx_next, vy_next


# 障害物が存在する場合
class Boid_obs:
    # 諸定数
    def __init__(self, num, r_s, r_a, r_c, r_o, w_s, w_a, w_c, w_o, lx, ly, obs):
        self.num = num
        self.r_s = r_s
        self.r_a = r_a
        self.r_c = r_c
        self.r_o = r_o
        self.w_s = w_s
        self.w_a = w_a
        self.w_c = w_c
        self.w_o = w_o
        self.lx = lx
        self.ly = ly
        self.obs = obs

        # 時間刻みはば
        self.dt = 0.1
        # 固定最大速度
        self.v_max = 3.0

    # 周期境界条件
    def pbc(self, x, y):
        output_x = np.where(x < 0, x + self.lx, np.where(self.lx < x, x - self.lx, x))
        output_y = np.where(y < 0, y + self.ly, np.where(self.ly < y, y - self.ly, y))
        return output_x, output_y
    
    # 周期境界条件下での距離行列の計算
    def pbc_ad(self, dx, dy, range):
        #dx_sing = dx > 0
        #dy_sing = dy > 0
        dx_abs = np.abs(dx)
        dy_abs = np.abs(dy)
        dx_abs_inv = np.abs(self.lx - dx_abs)
        dy_abs_inv = np.abs(self.ly - dy_abs)

        index_x = dx_abs > dx_abs_inv
        index_y = dy_abs > dy_abs_inv
        #dx_abs_true = np.where(index_x == True, dx_abs_inv, dx_abs)
        #dy_abs_true = np.where(index_y == True, dy_abs_inv, dy_abs)
        dx = np.where(index_x == True, -dx_abs_inv, dx)
        dy = np.where(index_y == True, -dy_abs_inv, dy)
        dist = np.sqrt(dx**2 + dy**2)
        ad_matrix = dist < range
        return dx, dy, dist, ad_matrix

    # 初期位置と初期速度の生成
    def reset(self):
        x = np.random.uniform(high=self.lx, low=0, size=self.num)
        y = np.random.uniform(high=self.lx, low=0, size=self.num)
        vx = np.random.uniform(size=self.num) * 0.1
        vy = np.random.uniform(size=self.num) * 0.1
        return x, y, vx, vy
    
    # 正規化計算式
    def normalize(self, dx, dy):
        norm = np.sqrt(dx**2 + dy**2)
        dx_normal = np.where(norm != 0, dx / norm, 0)
        dy_normal = np.where(norm != 0, dy / norm, 0)
        return dx_normal, dy_normal
    
    # 隣接行列らの計算
    def neighbor(self, x, y, range):
        dx = x[np.newaxis,:] - x[:,np.newaxis] # x_i - x_j, i is me
        dy = y[np.newaxis,:] - y[:,np.newaxis]
        #dist = self.pbc_dist(np.sqrt(dx**2 + dy**2))
        #neighbor_index = dist < range
        #ad_matrix = neighbor_index - np.eye(self.num)

        dx, dy, dist, ad_matrix = self.pbc_ad(dx, dy, range)
        return dx, dy, dist, ad_matrix
    
    # 分離の計算
    def separate(self, x, y):
        dx, dy, dist, ad_matrix = self.neighbor(x, y, self.r_s)
        dist_neighbor = np.sum(dist * ad_matrix, axis=1)
        dx_neighbor = np.sum(dx * ad_matrix, axis=1)
        dy_neighbor = np.sum(dy * ad_matrix, axis=1)

        fx = np.where(dist_neighbor != 0, dx_neighbor / dist_neighbor, 0)
        fy = np.where(dist_neighbor != 0, dy_neighbor / dist_neighbor, 0)
        return -fx, -fy
    
    # 整列の計算
    def alignment(self, x, y, vx, vy):
        dx, dy, dist, ad_matrix = self.neighbor(x, y, self.r_a)
        num_neighbor = np.sum(ad_matrix, axis=1)
        vx_neighbor = vx[:,np.newaxis] * ad_matrix
        vy_neighbor = vy[:,np.newaxis] * ad_matrix

        vx_a = np.where(num_neighbor != 0, np.sum(vx_neighbor, axis=1) / num_neighbor, 0)
        vy_a = np.where(num_neighbor != 0, np.sum(vy_neighbor, axis=1) / num_neighbor, 0)
        fx = vx_a - vx
        fy = vy_a - vy
        return fx, fy
    
    # 結合の計算
    def cohesion(self, x, y):
        dx, dy, dist, ad_matrix = self.neighbor(x, y, self.r_c)
        num_neighbor = np.sum(ad_matrix, axis=1)
        dx_neighbor = np.sum(dx * ad_matrix, axis=1)
        dy_neighbor = np.sum(dy * ad_matrix, axis=1)

        fx = np.where(num_neighbor != 0, dx_neighbor / num_neighbor, 0)
        fy = np.where(num_neighbor != 0, dy_neighbor / num_neighbor, 0)
        return fx, fy
    
    # 障害物からの斥力
    def obstacles(self, x, y):
        obs_xy = self.obs.T
        x_o, y_o = obs_xy[0], obs_xy[1]
        dx = x_o[np.newaxis,:] - x[:,np.newaxis] # obs - me
        dy = y_o[np.newaxis,:] - y[:,np.newaxis]
        dist = np.sqrt(dx**2 + dy**2)
        neighbor_obs_index = dist < self.r_o
        num_neighbor_obs = np.sum(neighbor_obs_index, axis=1)

        dx_neighbor = np.sum(dx * neighbor_obs_index, axis=1)
        dy_neighbor = np.sum(dy * neighbor_obs_index, axis=1)
        fx = np.where(num_neighbor_obs == 0, 0, dx_neighbor / num_neighbor_obs)
        fy = np.where(num_neighbor_obs == 0, 0, dy_neighbor / num_neighbor_obs)
        return -fx, -fy
    
    # 時間を進める
    def step(self, x, y, vx, vy):
        fx_s, fy_s = self.separate(x, y)
        fx_a, fy_a = self.alignment(x, y, vx, vy)
        fx_c, fy_c = self.cohesion(x, y)
        fx_o, fy_o = self.obstacles(x, y)

        # 正規化
        fx_s_normal, fy_s_normal = self.normalize(fx_s, fy_s)
        fx_a_normal, fy_a_normal = self.normalize(fx_a, fy_a)
        fx_c_normal, fy_c_normal = self.normalize(fx_c, fy_c)
        fx_o_normal, fy_o_normal = self.normalize(fx_o, fy_o)
        
        # f = ma
        fx = self.w_s * fx_s_normal + self.w_a * fx_a_normal + self.w_c * fx_c_normal + self.w_o * fx_o_normal
        fy = self.w_s * fy_s_normal + self.w_a * fy_a_normal + self.w_c * fy_c_normal + self.w_o * fy_o_normal

        # 速度の更新：v_t+1 = v_t + f * dt
        vx_next = vx + fx * self.dt
        vy_next = vy + fy * self.dt
        v_next = np.sqrt(vx_next**2 + vy_next**2)
        vx_next = np.where(v_next > self.v_max, vx_next / v_next * self.v_max, vx_next)
        vy_next = np.where(v_next > self.v_max, vy_next / v_next * self.v_max, vy_next)

        # 位置の更新：p_t+1 = p_t + v_t+1 * dt
        x_next = x + vx_next * self.dt
        y_next = y + vy_next * self.dt
        x_next_pbc, y_next_pbc = self.pbc(x_next, y_next)
        return x_next_pbc, y_next_pbc, vx_next, vy_next
    

    # 障害物が存在する場合
class Boid_chasing:
    # 諸定数
    def __init__(self, num, r_s, r_a, r_c, r_o, w_s, w_a, w_c, w_o, lx, ly, num_chaser, r_chaser, w_chaser):
        self.num = num
        self.r_s = r_s
        self.r_a = r_a
        self.r_c = r_c
        self.r_o = r_o
        self.w_s = w_s
        self.w_a = w_a
        self.w_c = w_c
        self.w_o = w_o
        self.lx = lx
        self.ly = ly
        self.num_chaser = num_chaser
        self.r_chaser = r_chaser
        self.w_chaser = w_chaser

        # 時間刻みはば
        self.dt = 0.1
        # 固定最大速度
        self.v_max = 3.0

    # 周期境界条件
    def pbc(self, x, y):
        output_x = np.where(x < 0, x + self.lx, np.where(self.lx < x, x - self.lx, x))
        output_y = np.where(y < 0, y + self.ly, np.where(self.ly < y, y - self.ly, y))
        return output_x, output_y
    
    # 周期境界条件下での距離行列の計算
    def pbc_ad(self, dx, dy, range):
        #dx_sing = dx > 0
        #dy_sing = dy > 0
        dx_abs = np.abs(dx)
        dy_abs = np.abs(dy)
        dx_abs_inv = np.abs(self.lx - dx_abs)
        dy_abs_inv = np.abs(self.ly - dy_abs)

        index_x = dx_abs > dx_abs_inv
        index_y = dy_abs > dy_abs_inv
        #dx_abs_true = np.where(index_x == True, dx_abs_inv, dx_abs)
        #dy_abs_true = np.where(index_y == True, dy_abs_inv, dy_abs)
        dx = np.where(index_x == True, -dx_abs_inv, dx)
        dy = np.where(index_y == True, -dy_abs_inv, dy)
        dist = np.sqrt(dx**2 + dy**2)
        ad_matrix = dist < range
        return dx, dy, dist, ad_matrix

    # 初期位置と初期速度の生成
    def reset(self):
        x = np.random.uniform(high=self.lx, low=0, size=self.num)
        y = np.random.uniform(high=self.lx, low=0, size=self.num)
        vx = np.random.uniform(size=self.num) * 0.1
        vy = np.random.uniform(size=self.num) * 0.1
        return x, y, vx, vy
    
    # 追跡者の初期位置と初期速度の生成
    def reset_chaser(self):
        x = np.random.uniform(high=self.lx, low=0, size=self.num_chaser)
        y = np.random.uniform(high=self.lx, low=0, size=self.num_chaser)
        vx = np.random.uniform(size=self.num_chaser) * 0.1
        vy = np.random.uniform(size=self.num_chaser) * 0.1
        return x, y, vx, vy
    
    # 正規化計算式
    def normalize(self, dx, dy):
        norm = np.sqrt(dx**2 + dy**2)
        dx_normal = np.where(norm != 0, dx / norm, 0)
        dy_normal = np.where(norm != 0, dy / norm, 0)
        return dx_normal, dy_normal
    
    # 隣接行列らの計算
    def neighbor(self, x, y, range):
        dx = x[np.newaxis,:] - x[:,np.newaxis] # x_i - x_j, i is me
        dy = y[np.newaxis,:] - y[:,np.newaxis]
        #dist = self.pbc_dist(np.sqrt(dx**2 + dy**2))
        #neighbor_index = dist < range
        #ad_matrix = neighbor_index - np.eye(self.num)

        dx, dy, dist, ad_matrix = self.pbc_ad(dx, dy, range)
        return dx, dy, dist, ad_matrix
    
    # 分離の計算
    def separate(self, x, y):
        dx, dy, dist, ad_matrix = self.neighbor(x, y, self.r_s)
        dist_neighbor = np.sum(dist * ad_matrix, axis=1)
        dx_neighbor = np.sum(dx * ad_matrix, axis=1)
        dy_neighbor = np.sum(dy * ad_matrix, axis=1)

        fx = np.where(dist_neighbor != 0, dx_neighbor / dist_neighbor, 0)
        fy = np.where(dist_neighbor != 0, dy_neighbor / dist_neighbor, 0)
        return -fx, -fy
    
    # 整列の計算
    def alignment(self, x, y, vx, vy):
        dx, dy, dist, ad_matrix = self.neighbor(x, y, self.r_a)
        num_neighbor = np.sum(ad_matrix, axis=1)
        vx_neighbor = vx[:,np.newaxis] * ad_matrix
        vy_neighbor = vy[:,np.newaxis] * ad_matrix

        vx_a = np.where(num_neighbor != 0, np.sum(vx_neighbor, axis=1) / num_neighbor, 0)
        vy_a = np.where(num_neighbor != 0, np.sum(vy_neighbor, axis=1) / num_neighbor, 0)
        fx = vx_a - vx
        fy = vy_a - vy
        return fx, fy
    
    # 結合の計算
    def cohesion(self, x, y):
        dx, dy, dist, ad_matrix = self.neighbor(x, y, self.r_c)
        num_neighbor = np.sum(ad_matrix, axis=1)
        dx_neighbor = np.sum(dx * ad_matrix, axis=1)
        dy_neighbor = np.sum(dy * ad_matrix, axis=1)

        fx = np.where(num_neighbor != 0, dx_neighbor / num_neighbor, 0)
        fy = np.where(num_neighbor != 0, dy_neighbor / num_neighbor, 0)
        return fx, fy
    
    # 追跡者からの斥力
    def obstacles(self, x, y, x_c, y_c):
        dx = x_c[np.newaxis,:] - x[:,np.newaxis] # obs - me
        dy = y_c[np.newaxis,:] - y[:,np.newaxis]
        dist = np.sqrt(dx**2 + dy**2)
        neighbor_obs_index = dist < self.r_o
        num_neighbor_obs = np.sum(neighbor_obs_index, axis=1)

        dx_neighbor = np.sum(dx * neighbor_obs_index, axis=1)
        dy_neighbor = np.sum(dy * neighbor_obs_index, axis=1)
        fx = np.where(num_neighbor_obs == 0, 0, dx_neighbor / num_neighbor_obs)
        fy = np.where(num_neighbor_obs == 0, 0, dy_neighbor / num_neighbor_obs)
        return -fx, -fy
    
    # 追跡者の駆動力
    def chaser_driven(self, x, y, x_c, y_c):
        dx = x_c[np.newaxis,:] - x[:,np.newaxis] # chaser - prey
        dy = y_c[np.newaxis,:] - y[:,np.newaxis]
        dist = np.sqrt(dx**2 + dy**2)
        neighbor_prey_index = dist < self.r_chaser
        num_neighbor_prey = np.sum(neighbor_prey_index, axis=0)

        dx_neighbor = np.sum(dx * neighbor_prey_index, axis=0)
        dy_neighbor = np.sum(dy * neighbor_prey_index, axis=0)
        fx = np.where(num_neighbor_prey == 0, 0, dx_neighbor / num_neighbor_prey)
        fy = np.where(num_neighbor_prey == 0, 0, dy_neighbor / num_neighbor_prey)
        return -fx, -fy
    
    # 時間を進める
    def step(self, x, y, vx, vy, x_c, y_c, vx_chaser, vy_chaser):
        fx_s, fy_s = self.separate(x, y)
        fx_a, fy_a = self.alignment(x, y, vx, vy)
        fx_c, fy_c = self.cohesion(x, y)
        fx_o, fy_o = self.obstacles(x, y, x_c, y_c)
        fx_chaser, fy_chaser = self.chaser_driven(x, y, x_c, y_c)

        # 正規化
        fx_s_normal, fy_s_normal = self.normalize(fx_s, fy_s)
        fx_a_normal, fy_a_normal = self.normalize(fx_a, fy_a)
        fx_c_normal, fy_c_normal = self.normalize(fx_c, fy_c)
        fx_o_normal, fy_o_normal = self.normalize(fx_o, fy_o)
        fx_chaser_normal, fy_chaser_normal = self.normalize(fx_chaser, fy_chaser)
        
        # f = ma
        fx = self.w_s * fx_s_normal + self.w_a * fx_a_normal + self.w_c * fx_c_normal + self.w_o * fx_o_normal
        fy = self.w_s * fy_s_normal + self.w_a * fy_a_normal + self.w_c * fy_c_normal + self.w_o * fy_o_normal
        fx_chaser = self.w_chaser * fx_chaser_normal
        fy_chaser = self.w_chaser * fy_chaser_normal

        # 速度の更新：v_t+1 = v_t + f * dt
        vx_next = vx + fx * self.dt
        vy_next = vy + fy * self.dt
        v_next = np.sqrt(vx_next**2 + vy_next**2)
        vx_next = np.where(v_next > self.v_max, vx_next / v_next * self.v_max, vx_next)
        vy_next = np.where(v_next > self.v_max, vy_next / v_next * self.v_max, vy_next)
        vx_chaser_next = vx_chaser + fx_chaser * self.dt
        vy_chaser_next = vy_chaser + fy_chaser * self.dt

        # 位置の更新：p_t+1 = p_t + v_t+1 * dt
        x_next = x + vx_next * self.dt
        y_next = y + vy_next * self.dt
        x_next_pbc, y_next_pbc = self.pbc(x_next, y_next)
        x_c_next = x_c + vx_chaser_next * self.dt
        y_c_next = y_c + vy_chaser_next * self.dt
        x_c_next_pbc, y_c_next_pbc = self.pbc(x_c_next, y_c_next)
        return x_next_pbc, y_next_pbc, vx_next, vy_next, x_c_next_pbc, y_c_next_pbc, vx_chaser_next, vy_chaser_next



#%%
# シミュレーター
"""
 ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄  ▄▄       ▄▄  ▄         ▄  ▄            ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄ 
▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░▌     ▐░░▌▐░▌       ▐░▌▐░▌          ▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌
▐░█▀▀▀▀▀▀▀▀▀  ▀▀▀▀█░█▀▀▀▀ ▐░▌░▌   ▐░▐░▌▐░▌       ▐░▌▐░▌          ▐░█▀▀▀▀▀▀▀█░▌ ▀▀▀▀█░█▀▀▀▀ ▐░█▀▀▀▀▀▀▀█░▌▐░█▀▀▀▀▀▀▀█░▌
▐░▌               ▐░▌     ▐░▌▐░▌ ▐░▌▐░▌▐░▌       ▐░▌▐░▌          ▐░▌       ▐░▌     ▐░▌     ▐░▌       ▐░▌▐░▌       ▐░▌
▐░█▄▄▄▄▄▄▄▄▄      ▐░▌     ▐░▌ ▐░▐░▌ ▐░▌▐░▌       ▐░▌▐░▌          ▐░█▄▄▄▄▄▄▄█░▌     ▐░▌     ▐░▌       ▐░▌▐░█▄▄▄▄▄▄▄█░▌
▐░░░░░░░░░░░▌     ▐░▌     ▐░▌  ▐░▌  ▐░▌▐░▌       ▐░▌▐░▌          ▐░░░░░░░░░░░▌     ▐░▌     ▐░▌       ▐░▌▐░░░░░░░░░░░▌
 ▀▀▀▀▀▀▀▀▀█░▌     ▐░▌     ▐░▌   ▀   ▐░▌▐░▌       ▐░▌▐░▌          ▐░█▀▀▀▀▀▀▀█░▌     ▐░▌     ▐░▌       ▐░▌▐░█▀▀▀▀█░█▀▀ 
          ▐░▌     ▐░▌     ▐░▌       ▐░▌▐░▌       ▐░▌▐░▌          ▐░▌       ▐░▌     ▐░▌     ▐░▌       ▐░▌▐░▌     ▐░▌  
 ▄▄▄▄▄▄▄▄▄█░▌ ▄▄▄▄█░█▄▄▄▄ ▐░▌       ▐░▌▐░█▄▄▄▄▄▄▄█░▌▐░█▄▄▄▄▄▄▄▄▄ ▐░▌       ▐░▌     ▐░▌     ▐░█▄▄▄▄▄▄▄█░▌▐░▌      ▐░▌ 
▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░▌       ▐░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░▌       ▐░▌     ▐░▌     ▐░░░░░░░░░░░▌▐░▌       ▐░▌
 ▀▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀▀  ▀         ▀  ▀▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀▀  ▀         ▀       ▀       ▀▀▀▀▀▀▀▀▀▀▀  ▀         ▀ 
"""
import numpy as np

class Simulator:
    # 諸定数
    def __init__(self, num, r_s, r_a, r_c, w_s, w_a, w_c, lx, ly, Tmax):
        self.num = num
        self.r_s = r_s
        self.r_a = r_a
        self.r_c = r_c
        self.w_s = w_s
        self.w_a = w_a
        self.w_c = w_c
        self.lx = lx
        self.ly = ly
        self.Tmax = Tmax

        # 時間刻みはば（固定）
        self.dt = 0.1

    # シミュレーション
    def run(self):
        agent = Boid(self.num, self.r_s, self.r_a, self.r_c, self.w_s, self.w_a, self.w_c, self.lx, self.ly)
        x, y, vx, vy = agent.reset()

        x_rec = [x]
        y_rec = [y]
        vx_rec = [vx]
        vy_rec = [vy]

        for t in range(int(self.Tmax / self.dt)):
            x_next, y_next, vx_next, vy_next = agent.step(x, y, vx, vy)
            x, y, vx, vy = x_next, y_next, vx_next, vy_next

            x_rec.append(x)
            y_rec.append(y)
            vx_rec.append(vx)
            vy_rec.append(vy)

            progress = (t+1) / (self.Tmax / self.dt) * 100
            print('Progress {:.3f}%'.format(progress), end="\r")
        return x_rec, y_rec, vx_rec, vy_rec
    
    # シミュレーション
    def run_obs(self, r_o, w_o, obs):
        agent = Boid_obs(self.num, self.r_s, self.r_a, self.r_c, r_o, self.w_s, self.w_a, self.w_c, w_o, self.lx, self.ly, obs)
        x, y, vx, vy = agent.reset()

        x_rec = [x]
        y_rec = [y]
        vx_rec = [vx]
        vy_rec = [vy]

        for t in range(int(self.Tmax / self.dt)):
            x_next, y_next, vx_next, vy_next = agent.step(x, y, vx, vy)
            x, y, vx, vy = x_next, y_next, vx_next, vy_next

            x_rec.append(x)
            y_rec.append(y)
            vx_rec.append(vx)
            vy_rec.append(vy)

            progress = (t+1) / (self.Tmax / self.dt) * 100
            print('Progress {:.3f}%'.format(progress), end="\r")
        return x_rec, y_rec, vx_rec, vy_rec
    
    # シミュレーション
    def run_chaser(self, r_o, w_o, r_chaser, w_chaser, num_chaser):
        agent = Boid_chasing(self.num, self.r_s, self.r_a, self.r_c, r_o, self.w_s, self.w_a, self.w_c, w_o, self.lx, self.ly, num_chaser, r_chaser, w_chaser )
        x, y, vx, vy = agent.reset()
        x_c, y_c, vx_c, vy_c = agent.reset_chaser()

        x_rec = [x]
        y_rec = [y]
        vx_rec = [vx]
        vy_rec = [vy]

        xc_rec = [x_c]
        yc_rec = [y_c]
        vxc_rec = [vx_c]
        vyc_rec = [vy_c]

        for t in range(int(self.Tmax / self.dt)):
            x_next, y_next, vx_next, vy_next, x_c_next, y_c_next, vx_c_next, vy_c_next = agent.step(x, y, vx, vy, x_c, y_c, vx_c, vy_c)
            x, y, vx, vy = x_next, y_next, vx_next, vy_next
            x_c, y_c, vx_c, vy_c = x_c_next, y_c_next, vx_c_next, vy_c_next

            x_rec.append(x)
            y_rec.append(y)
            vx_rec.append(vx)
            vy_rec.append(vy)

            xc_rec.append(x_c)
            yc_rec.append(y_c)
            vxc_rec.append(vx_c)
            vyc_rec.append(vy_c)

            progress = (t+1) / (self.Tmax / self.dt) * 100
            print('Progress {:.3f}%'.format(progress), end="\r")
        return x_rec, y_rec, vx_rec, vy_rec, xc_rec, yc_rec, vxc_rec, vyc_rec


#%%
# 平均が0、標準偏差が1のガウス分布から乱数を生成
random_numbers = np.random.normal(loc=0, scale=100, size=1000)
#random_numbers = np.random.uniform(high=0.0, low=-2.0, size=1000)

# 乱数の最大値と最小値を取得
min_val = np.min(random_numbers)
max_val = np.max(random_numbers)

# 乱数を-1から1の範囲にスケーリング
scaled_random_numbers = ((random_numbers - min_val) / (max_val - min_val)) * 4 - 2
#np.save('random/No13', scaled_random_numbers)

# シミュレーションを実行
num = 1000 # 個体数
r_s = 1.0 # 分離性半径
r_a = 1.0 # 整列性半径
r_c = 1.0 # 集合性半径
r_o = 1.0 # 斥力性半径
w_s = 1.0 # 分離性
w_a = 1.0 # 整列性
w_c = 1.0 # 集合性
w_o = 1.0 # 障害物からの斥力性
lx = 10 # シミュレーション領域
ly = 10
Tmax = 100 # シミュレーション継続時間

dt = 0.1 # 固定の時間刻み幅

obs = np.array([[3,3], [7,7], [3,7], [7,3]]) # 障害物の[x,y]座標

num_chaser = 1 # 追跡者の個体数
r_chaser = 1.0 # 追跡者の視界半径
w_chaser = 1.0 # 追跡者の駆動力の重み

data = np.array([num, r_s, r_a, r_c, w_s, w_a, w_c, lx, ly, Tmax])
#data = np.array([num, r_s, r_a, r_c, r_o, w_s, w_a, w_c, w_o, lx, ly, obs.flatten, Tmax])
#data = np.array([num, r_s, r_a, r_c, w_s, w_a, w_c, lx, ly, Tmax, num_chaser, r_chaser, w_chaser])
np.save('parameter/No13', data)

machine = Simulator(num, r_s, r_a, r_c, w_s, w_a, w_c, lx, ly, Tmax)
x, y, vx, vy = machine.run()
record = np.array([x, y, vx, vy])
np.save('record/No13', record)
#x, y, vx, vy = machine.run_obs(r_o, w_o, obs)
#x, y, vx, vy, x_c, y_c, vx_c, vy_c = machine.run_chaser(r_o, w_o, r_chaser, w_chaser, num_chaser)


#%%
# アニメーションを作成
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML

# キャンバスを設置
fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot()

ax1.set_xlim(0, lx)
ax1.set_ylim(0, ly)

# 障害物の座標
obs_x = obs.T[0]
obs_y = obs.T[1]

#アップデート関数
def update1(i):
    ax1.clear()
    ax1.set_xlim(0, lx)
    ax1.set_ylim(0, ly)

    data_x, data_y, data_vx, data_vy = x[i], y[i], vx[i], vy[i]
    theta = np.arctan2(data_vy, data_vx)
    order = np.sqrt(np.sum(np.cos(theta))**2 + np.sum(np.sin(theta))**2) / num
    #data_x_c, data_y_c, data_vx_c, data_vy_c = x_c[i], y_c[i], vx_c[i], vy_c[i]

    #ax1.plot(obs_x, obs_y, 's', color='red') # 障害物の表示

    #ax1.scatter(data_x_c, data_y_c, color='red')
    #ax1.scatter(data_x, data_y, color='black')
    ax1.quiver(data_x, data_y, np.cos(theta), np.sin(theta), color='blue')
    #ax1.set_title('time={:.1f}'.format((i+1)*dt), fontsize=14)
    ax1.set_title('num={}, r_s={}, r_a={}, r_c={}, w_s={}, w_a={}, w_c={}, time={:.2f}, order={:.2f}'.format(num, r_s, r_a, r_c, w_s, w_a, w_c, (i+1)*dt, order))
    # 進捗状況を出力する
    progress_rate = (i+1)/(Tmax/dt)*100
    print("Animation progress={:.3f}%".format(progress_rate), end='\r')
        
#アニメーション作成とgif保存
ani = animation.FuncAnimation(fig, update1, frames=range(int(Tmax/dt)))
    
#グラフ表示
plt.show()

#アニメーションを表示
HTML(ani.to_jshtml())

# アニメーションの保存
ani.save('animation/data=No13_quiver.mp4', writer='ffmpeg')


#%%
# 秩序変数のグラフ
import matplotlib.pyplot as plt
x, y, vx, vy = np.load('record/No13.npy')
theta = np.arctan2(vy, vx)
order = np.sqrt(np.sum(np.cos(theta), axis=1)**2 + np.sum(np.sin(theta), axis=1)**2) / num

plt.plot(np.linspace(1, 100, 1000), order[1:],  '-')
plt.xlabel('Time')
plt.ylabel('Order parameter')
plt.title('Used record = No13')
plt.grid(True)
plt.savefig('graph/orderNo13.png')
plt.show()


# %%
# 分布のヒストグラムの生成
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib

# ヒストグラムのプロット
plt.hist(w_s + scaled_random_numbers, bins=30, color='blue', alpha=0.7)
plt.xlabel('値')
plt.ylabel('頻度')
plt.title('w_sの分布')
plt.grid(True)
#plt.savefig('graph/distNo13w_s.png') # plt.show()の前に保存を行う
plt.show()


# %%
# 数値を変えて試してみる
import numpy as np

class Test:
    def __init__(self):
        self.dw = 0.01
        self.num_test = int(1/self.dw)

    def w_s(self, data):
        num, r_s, r_a, r_c, w_s, w_a, w_c, lx, ly, Tmax = int(data[0]), data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9]

        orders = []
        w_s_s = []
        for i in range(self.num_test + 1):
            dw = self.dw * i
            w_s_s.append(w_s + dw)
            machine = Simulator(num, r_s, r_a, r_c, w_s + dw, w_a, w_c, lx, ly, Tmax)
            x, y, vx, vy = machine.run()
            theta = np.arctan2(vy, vx)
            order = np.sqrt(np.sum(np.cos(theta), axis=1)**2 + np.sum(np.sin(theta), axis=1)**2) / num
            orders.append(order)
            print('w_s={}の収束値: order parameter={:.3f}'.format(w_s + dw, np.mean(order[int(Tmax/2):])))
        return orders, w_s_s
    

#%%
# 実際にシミュレーションをする
import numpy as np
data = np.load('parameter/No13.npy')
test_machine = Test()
order_data, w_s_data = test_machine.w_s(data)


#%%
# テスト結果をグラフで描画する
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib

for i in range(11):
    use_order = order_data[i]
    use_weight = w_s_data[i]
    x = np.linspace(1, 100, 1000)
    plt.plot(x, use_order[1:], alpha=0.4, color='blue')
    plt.text(x[-1]+1, use_order[-1], '{:.2f}'.format(use_weight), fontsize=8)
plt.xlabel('Time')
plt.ylabel('Order parameter')
plt.title('各w_sの値における秩序変数の時間推移')
plt.grid(True)
plt.savefig('graph/秩序変数推移w_s_001.png')
plt.show()

#%%
order_data = np.array(order_data)
order_conv = np.mean(order_data, axis=1)
plt.plot(np.linspace(1.0, 2.0, 101), order_conv, 'ro-')
plt.ylabel('Order parameterの収束値')
plt.xlabel('w_s値')
plt.title('各w_sの値における秩序変数の収束値')
plt.grid(True)
plt.savefig('graph/秩序変数収束値w_s_001.png')
plt.show()
#%%
import numpy as np

datas = np.load('parameter/No13.npy')
print(datas)
print(len(data))
# %%
