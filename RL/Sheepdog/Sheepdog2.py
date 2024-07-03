
# 参考文献：https://cattech-lab.com/science-tools/sheepdog-simulation/#ref1
# 牧羊犬の制御モデルを実装

#%%
# 羊の制御モデルの実装
import numpy as np
import random

class Sheep:
    def __init__(self, num_sheep, d_ss1, d_ss2, d_gr, d_dg, V_max1, V_max2, w0, w1, w2_1, w2_2, w3_1, w3_2, lx, ly):
        self.N = num_sheep
        self.d_ss1 = d_ss1 # 犬がいる
        self.d_ss2 = d_ss2 # 犬がいない
        self.d_gr = d_gr
        self.d_dg = d_dg
        self.V_max1 = V_max1 # 牧羊犬がいる
        self.V_max2 = V_max2 # 牧羊犬がいない
        self.w0 = w0
        self.w1 = w1
        self.w2_1 = w2_1 # 牧羊犬がいる
        self.w2_2 = w2_2
        self.w3_1 = w3_1 # 牧羊犬がいる
        self.w3_2 = w3_2
        self.lx = lx
        self.ly = ly
        
    def calculate_neighbor(self, x, y):
        dx = x[np.newaxis, :] - x[:, np.newaxis]
        dy = y[np.newaxis, :] - y[:, np.newaxis]
        dist = np.sqrt(dx**2 + dy**2)
        return dx, dy, dist
    
    def calculate_neighbor_sheep(self, x_s, y_s, x_d, y_d):
        dx = x_s - x_d
        dy = y_s - y_d
        dist = np.sqrt(dx**2 + dy**2)
        return dx, dy, dist
    
    def vector_normalize(self, x, y):
        xy = np.column_stack((x, y))
        xy_norm = np.linalg.norm(xy, axis=1)
        # Avoid division by zero by checking if the norm is non-zero
        xy_normalize = np.divide(xy, xy_norm[:, None], out=np.zeros_like(xy, dtype=float), where=(xy_norm[:, None] != 0))
        return xy_normalize[:, 0], xy_normalize[:, 1]
    
    def BC(self, x, y, u_x, u_y):
        output_u_x = []
        output_u_y = []
        for i in range(self.N):
            if np.abs(x[i] - 0) < 3 or np.abs(x[i] - self.lx) < 3:
                if np.abs(x[i] - 0) < 3 or np.abs(x[i] - self.lx) < 3:
                    output_u_x_i = (self.lx/2 - x[i]) / np.abs(self.lx/2 - x[i]) * 3.0
                elif x[i] < 0 or x[i] > self.lx:
                    output_u_x_i = (self.lx/2 - x[i]) / np.abs(self.lx/2 - x[i]) * 5.0
            else:
                output_u_x_i = u_x[i]
            if np.abs(y[i] - 0) < 3 or np.abs(y[i] - self.ly) < 3:
                if np.abs(y[i] - 0) < 3 or np.abs(y[i] - self.lx) < 3:
                    output_u_y_i = (self.ly/2 - y[i]) / np.abs(self.ly/2 - y[i]) * 3.0
                elif y[i] < 0 or y[i] > self.ly:
                    output_u_y_i = (self.ly/2 - y[i]) / np.abs(self.ly/2 - y[i]) * 5.0
            else:
                output_u_y_i = u_y[i]
            output_u_x.append(output_u_x_i)
            output_u_y.append(output_u_y_i)
        return np.array(output_u_x), np.array(output_u_y)
    
    def BC_c(self, x, y, u_x, u_y):
        dx = x - self.lx/2
        dy = y - self.ly/2
        dist = np.sqrt(dx**2 + dy**2)
        output_x = []
        output_y = []
        for i in range(self.N):
            dist_i = dist[i]
            output_x_i = u_x[i] -2*(dist_i / self.lx/2) * u_x[i]
            output_y_i = u_y[i] -2*(dist_i / self.ly/2) * u_y[i]
            output_x.append(output_x_i)
            output_y.append(output_y_i)
        return np.array(output_x), np.array(output_y)
        
    def f1(self, x_sheep, y_sheep, x_dog, y_dog):
        dx, dy, dist = self.calculate_neighbor(x_sheep, y_sheep)
        dx_neighbor_dog, dy_neighbor_dog, dist_neighbor_dog = self.calculate_neighbor_sheep(x_sheep, y_sheep, x_dog, y_dog)
        #print('dist_neighbot_dog, {}'.format(dist_neighbor_dog))
        sheep_insight_dog = dist_neighbor_dog < self.d_dg
        #print('sheep_insight_dog, {}'.format(sheep_insight_dog))
        d_ss = np.where(sheep_insight_dog == True, self.d_ss1, self.d_ss2)
        #print('d_ss, {}'.format(d_ss))
        dist_neighbor = np.where(dist < d_ss, dist, 0)
        #print('dist_neighbor, {}'.format(dist_neighbor))
        dist_neighbor_norm = np.abs(dist_neighbor)
        #print('dist_neighbor_norm, {}'.format(dist_neighbor_norm))
        
        output_x = []
        output_y = []
        for i in range(self.N):
            if sheep_insight_dog[i] == True:
                d_ss_i = self.d_ss1
            else:
                d_ss_i = self.d_ss2
            dist_i = dist[i]
            dist_neighbor_i = np.where(dist_i < d_ss_i, dist_i, 0)
            dist_neighbor_norm_i = np.abs(dist_neighbor_i)
            dx_neighbor_i = np.where(dist_i < d_ss_i, dx[i], 0)
            dy_neighbor_i = np.where(dist_i < d_ss_i, dy[i], 0)
            output_x_i = np.nansum(-dx_neighbor_i) / (np.nansum((dist_neighbor_norm_i**2))) if np.nansum(dist_neighbor_norm_i) != 0 else 0
            output_y_i = np.nansum(-dy_neighbor_i) / (np.nansum((dist_neighbor_norm_i**2))) if np.nansum(dist_neighbor_norm_i) != 0 else 0
            output_x.append(output_x_i)
            output_y.append(output_y_i)
        
        #print('f1, {}, {}'.format(output_x, output_y))
        return np.array(output_x), np.array(output_y)
    
    # 要調査
    def f2(self, x, y):
        dx, dy, dist = self.calculate_neighbor(x, y)
        neighbor = dist < self.d_gr
        
        x_gravity = []
        y_gravity = []
        for i in range(self.N):
            neighbor_i = neighbor[i]
            x_neighbor_i = x[neighbor_i == True]
            y_neighbor_i = y[neighbor_i == True]
            x_gravity_i = np.nanmean(x_neighbor_i)
            y_gravity_i = np.nanmean(y_neighbor_i)
            x_gravity.append(x_gravity_i)
            y_gravity.append(y_gravity_i)
        
        output_x = np.array(x_gravity) - x
        output_y = np.array(y_gravity) - y
        #print('f2, {}, {}'.format(output_x, output_y))
        return output_x, output_y
    
    # 要調査
    def f3(self, x_sheep, y_sheep, x_dog, y_dog):
        dx = x_dog - x_sheep
        dy = y_dog - y_sheep
        dsit_dog_sheep = np.sqrt(dx**2 + dy**2) < self.d_dg
        
        output_x = np.where(dsit_dog_sheep == True, -dx, 0)
        output_y = np.where(dsit_dog_sheep == True, -dy, 0)
        #print('f3, {}, {}'.format(output_x, output_y))
        return output_x, output_y
    
    def reset(self):
        x = np.random.uniform(low=0 + 20, high=self.lx - 20, size=self.N)
        y = np.random.uniform(low=0 + 20, high=self.ly - 20, size=self.N)
        u_x = np.zeros(self.N)
        u_y = np.zeros(self.N)
        x_ = []
        y_ = []
        """
        n = int(self.N / 10)
        for i in range(n):
            cent_theta = np.random.uniform(low=np.pi/2 * (i+1), high=np.pi * (i+1))
            x_cent = 50 * np.cos(cent_theta) + self.lx/2
            y_cent = 50 * np.sin(cent_theta) + self.ly/2
            x = np.random.uniform(low=x_cent - 10, high=x_cent + 10, size=int(self.N/n))
            y = np.random.uniform(low=y_cent - 10, high=y_cent + 10, size=int(self.N/n))
            x_.append(x)
            y_.append(y)
        """
        return x, y, u_x, u_y
    
    def step(self, x_sheep, y_sheep, x_dog, y_dog, v_x, v_y):
        dx_neighbor_dog, dy_neighbor_dog, dist_neighbor_dog = self.calculate_neighbor_sheep(x_sheep, y_sheep, x_dog, y_dog)
        sheep_insight_dog = dist_neighbor_dog < self.d_dg
        
        v1_x, v1_y = self.f1(x_sheep, y_sheep, x_dog, y_dog)
        v2_x, v2_y = self.f2(x_sheep, y_sheep)
        v3_x, v3_y = self.f3(x_sheep, y_sheep, x_dog, y_dog)
        
        v1_x_normal, v1_y_normal = self.vector_normalize(v1_x, v1_y)
        v2_x_normal, v2_y_normal = self.vector_normalize(v2_x, v2_y)
        v3_x_normal, v3_y_normal = self.vector_normalize(v3_x, v3_y)
        v_x_normal, v_y_normal = self.vector_normalize(v_x, v_y)
        
        w2 = np.where(sheep_insight_dog == True, self.w2_1, self.w2_2)
        w3 = np.where(sheep_insight_dog == True, self.w3_1, self.w3_2)
        
        v_x_next = self.w1 * v1_x_normal + w2 * v2_x_normal + w3 * v3_x_normal
        v_y_next = self.w1 * v1_y_normal + w2 * v2_y_normal + w3 * v3_y_normal
        
        V_x_next = self.w0 * v_x_normal + v_x_next
        V_y_next = self.w0 * v_y_normal + v_y_next
        V_next_norm = np.linalg.norm(np.column_stack((V_x_next, V_y_next)), axis=1)
        V_x_e, V_y_e = self.vector_normalize(V_x_next, V_y_next)
        
        index_max1 = V_next_norm > self.V_max1
        index_max2 = V_next_norm > self.V_max2
        
        output_V_x_next = []
        output_V_y_next = []
        for i in range(self.N):
            V_x_next_i = V_x_next[i]
            V_y_next_i = V_y_next[i]
            if sheep_insight_dog[i] == True:
                if index_max1[i] == True:
                    output_V_x_next_i = self.V_max1 * V_x_e[i]
                    output_V_y_next_i = self.V_max1 * V_y_e[i]
                else:
                    output_V_x_next_i = V_x_next_i
                    output_V_y_next_i = V_y_next_i
            else:
                if index_max2[i] == True:
                    output_V_x_next_i = self.V_max2 * V_x_e[i]
                    output_V_y_next_i = self.V_max2 * V_y_e[i]
                else:
                    output_V_x_next_i = V_x_next_i
                    output_V_y_next_i = V_y_next_i
            output_V_x_next.append(output_V_x_next_i)
            output_V_y_next.append(output_V_y_next_i)
        output_V_x_next, output_V_y_next = self.BC(x_sheep, y_sheep, np.array(output_V_x_next), np.array(output_V_y_next))
        
        # output_V_x_next周辺でのバグを解決するべし ： 解決
        #print('output_V_x_next, {}'.format(output_V_x_next))
        output_x = x_sheep + output_V_x_next
        output_y = y_sheep + output_V_y_next
        
        return output_x, output_y, output_V_x_next, output_V_y_next
    

#%%
# 牧羊犬の制御モデルを実装
import numpy as np
import random
from scipy.spatial import ConvexHull

class Sheepdog:
    def __init__(self, num_sheep, d_sd, d_gr, d_fn, d_dr, d_cl, U_max, z0, z1, z2, x_start, y_start, x_target, y_target, lx, ly):
        self.N = num_sheep
        self.d_sd = d_sd
        self.d_gr = d_gr
        self.d_fn = d_fn
        self.d_dr = d_dr
        self.d_cl = d_cl
        self.U_max = U_max
        self.z0 = z0
        self.z1 = z1
        self.z2 = z2
        self.x_start = x_start
        self.y_start = y_start
        self.x_target = x_target
        self.y_target = y_target
        self.lx = lx
        self.ly = ly
        
    def calculate_neighbor_sheep(self, x_d, y_d, x_s, y_s):
        dx = x_s - x_d
        dy = y_s - y_d
        dist = np.sqrt(dx**2 + dy**2)
        return dx, dy, dist
    
    def vector_normalize(self, x, y):
        xy = np.column_stack((x, y))
        xy_norm = np.linalg.norm(xy, axis=1)
        # Avoid division by zero by checking if the norm is non-zero
        xy_normalize = np.divide(xy, xy_norm[:, None], out=np.zeros_like(xy, dtype=float), where=(xy_norm[:, None] != 0))
        return xy_normalize[:, 0], xy_normalize[:, 1]
    
    def BC(self, x, y, u_x, u_y):
        if np.abs(x - 0) < 5 or np.abs(x - self.lx) < 5:
            output_u_x = -u_x
        else:
            output_u_x = u_x
        if np.abs(y - 0) < 5 or np.abs(y - self.ly) < 5:
            output_u_y = -u_y
        else:
            output_u_y = u_y
        return np.array(output_u_x), np.array(output_u_y)
    
    def BC_c(self, x, y, u_x, u_y):
        dx = x - self.lx/2
        dy = y - self.ly/2
        dist = np.sqrt(dx**2 + dy**2)
        output_x = u_x -2*(dist / self.lx/2) * u_x
        output_y = u_y -2*(dist / self.ly/2) * u_y
        return output_x, output_y
        
    def f1(self, x_dog, y_dog, x_sheep, y_sheep):
        dx, dy, dist = self.calculate_neighbor_sheep(x_dog, y_dog, x_sheep, y_sheep)
        dist_insight = np.where(dist < self.d_sd, dist, self.d_sd + 1)
        most_neighbor_sheep_index = np.argmin(dist_insight)
        
        most_neighbor_dx = dx[most_neighbor_sheep_index]
        most_neighbor_dy = dy[most_neighbor_sheep_index]
        
        output_x = -most_neighbor_dx
        output_y = -most_neighbor_dy
        #print('f1_d, {}, {}'.format(output_x, output_y))
        return output_x, output_y

    def f2(self, x_dog, y_dog, x_sheep, y_sheep):
        dx, dy, dist_insight = self.calculate_neighbor_sheep(x_dog, y_dog, x_sheep, y_sheep)
        most_neighbor_sheep_index = np.argmin(dist_insight)
        most_neighbor_sheep_x = x_sheep[most_neighbor_sheep_index]
        most_neighbor_sheep_y = y_sheep[most_neighbor_sheep_index]
        
        dx_sheep_to_mns = most_neighbor_sheep_x - x_sheep
        dy_sheep_to_mns = most_neighbor_sheep_y - y_sheep
        dist_sheep_to_mns = np.sqrt(dx_sheep_to_mns**2 + dy_sheep_to_mns**2) < self.d_gr
        sheep_flock_x = x_sheep[dist_sheep_to_mns]
        sheep_flock_y = y_sheep[dist_sheep_to_mns]
        flock_gravity_x = np.nanmean(sheep_flock_x)
        flock_gravity_y = np.nanmean(sheep_flock_y)
        
        dx_g_to_side = flock_gravity_x - sheep_flock_x
        dy_g_to_side = flock_gravity_y - sheep_flock_y
        dist_g_to_side = np.sqrt(dx_g_to_side**2 + dy_g_to_side**2)
        condition_flocksize_dfn = dist_g_to_side < self.d_fn # 群れサイズがd_fnに収まっているかの条件
        
        # 群れサイズが条件を満たしていない場合
        if np.nansum(condition_flocksize_dfn) == 0:
            dx_most_far_sheep_to_g = x_sheep - flock_gravity_x
            dy_most_far_sheep_to_g = y_sheep - flock_gravity_y
            dist_most_far_sheep_to_g = np.sqrt(dx_most_far_sheep_to_g**2 + dy_most_far_sheep_to_g**2)
            most_far_sheep_index_from_flock = np.argmax(dist_most_far_sheep_to_g)
            
            most_far_sheep_x = x_sheep[most_far_sheep_index_from_flock]
            most_far_sheep_y = y_sheep[most_far_sheep_index_from_flock]
            
            m = most_far_sheep_index_from_flock + self.d_cl
            n = self.d_cl
            p_cl_x = (-n*flock_gravity_x + m*most_far_sheep_x) / (m - n)
            p_cl_y = (-n*flock_gravity_y + m*most_far_sheep_y) / (m - n)
            
            output_x = p_cl_x - x_dog
            output_y = p_cl_y - y_dog
        else:
            dx_g_to_t = flock_gravity_x - self.x_target
            dy_g_to_t = flock_gravity_y - self.y_target
            dist_g_to_t = np.sqrt(dx_g_to_t**2 + dy_g_to_t**2)
            
            m = dist_g_to_t + self.d_dr
            n = self.d_dr
            p_dr_x = (-n*self.x_target + m*flock_gravity_x) / (m - n)
            p_dr_y = (-n*self.y_target + m*flock_gravity_y) / (m - n)
            
            output_x = p_dr_x - x_dog
            output_y = p_dr_y - y_dog
        return output_x, output_y
    
    def f2_convex(self, x_dog, y_dog, x_sheep, y_sheep):
        p_sheep = np.column_stack((x_sheep, y_sheep))
        hull = ConvexHull(p_sheep)
        p_convex = hull.points
        p_convex_x = p_convex[:, 0]
        p_convex_y = p_convex[:, 1]
        
        #sheep_g_x = np.nanmean(x_sheep)
        #sheep_g_y = np.nanmean(y_sheep)
        
        """
        A = x_dog - self.x_target
        B = y_dog - self.y_target
        #d1 = np.abs(B * p_convex_x + (-A) * p_convex_y + (A * sheep_g_y - B * x_dog)) / np.sqrt(B**2 + A**2)
        d1 = np.abs(B * p_convex_x + (-A) * p_convex_y + (A * self.y_target - B * x_dog)) / np.sqrt(B**2 + A**2)
        dx2 = p_convex_x - x_dog
        dy2 = p_convex_y - y_dog
        d2 = np.sqrt(dx2**2 + dy2**2)
        d3 = np.sqrt((d1 - d2)*(d1 + d2))
        """
        
        vec_x_d_t = self.x_target - x_dog
        vec_y_d_t = self.y_target - y_dog
        vec_x_convex_d = p_convex_x - x_dog
        vec_y_convex_d = p_convex_y - y_dog
        d3 = []
        for i in range(self.N):
            vec_x_convex_d_i = vec_x_convex_d[i]
            vec_y_convex_d_i = vec_y_convex_d[i]
            vec_d_t_norm_power = np.sqrt(vec_x_d_t**2 + vec_y_d_t**2)**2
            dot_i = vec_x_d_t * vec_x_convex_d_i + vec_y_d_t * vec_y_convex_d_i
            projection_x = (dot_i / vec_d_t_norm_power) * vec_x_d_t
            projection_y = (dot_i / vec_d_t_norm_power) * vec_y_d_t
            d3_i = np.sqrt(projection_x**2 + projection_y**2)
            d3.append(d3_i)
        
        neighbor_index = np.argmin(np.array(d3))
        most_neighbor_p_convex_x = p_convex_x[neighbor_index]
        most_neighbor_p_convex_y = p_convex_y[neighbor_index]
        
        dx_convex_sheep = most_neighbor_p_convex_x - x_dog
        dy_convex_sheep = most_neighbor_p_convex_y - y_dog
        d_convex_sheep = np.sqrt(dx_convex_sheep**2 + dy_convex_sheep**2)
        
        theta = np.arctan(5/d_convex_sheep)
        thetas = np.array([theta, -theta])
        p_candidates = []
        p_candidates_d = []
        for i in range(len(thetas)):
            theta_i = thetas[i]
            p_candidate_i_x = most_neighbor_p_convex_x * np.cos(theta_i) - most_neighbor_p_convex_y * np.sin(theta)
            p_candidate_i_y = most_neighbor_p_convex_x * np.sin(theta_i) + most_neighbor_p_convex_y * np.cos(theta)
            dx_ = p_candidate_i_x - self.x_target
            dy_ = p_candidate_i_y - self.y_target
            dist_p_candidate_to_g = np.sqrt(dx_**2 + dy_**2)
            p_candidates_d.append(dist_p_candidate_to_g)
            p_candidate_i = np.array([p_candidate_i_x, p_candidate_i_y])
            p_candidates.append(p_candidate_i)
        condidate_index = np.argmin(np.array(p_candidates_d))
        p_candidate = p_candidates[condidate_index][:]
        
        output_x = p_candidate[0] - x_dog
        output_y = p_candidate[1] - y_dog
        return output_x, output_y
        
    def reset(self):
        x = self.x_start
        y = self.y_start
        u_x = 0
        u_y = 0
        return x, y, u_x, u_y
    
    def step(self, x_dog, y_dog, x_s, y_s, u_x, u_y):
        x_sheep = x_s[~np.isnan(x_s)]
        y_sheep = y_s[~np.isnan(y_s)]
        u1_x, u1_y = self.f1(x_dog, y_dog, x_sheep, y_sheep)
        u2_x, u2_y = self.f2(x_dog, y_dog, x_sheep, y_sheep)
        
        u1_x_normal, u1_y_normal = self.vector_normalize(u1_x, u1_y)
        u2_x_normal, u2_y_normal = self.vector_normalize(u2_x, u2_y)
        u_x_normal, u_y_normal = self.vector_normalize(u_x, u_y)
        
        u_x_next = self.z1 * u1_x_normal + self.z2 * u2_x_normal
        u_y_next = self.z1 * u1_y_normal + self.z2 * u2_y_normal
        
        U_x_next = self.z0 * u_x_normal + u_x_next
        U_y_next = self.z0 * u_y_normal + u_y_next
        U_next_norm = np.sqrt(U_x_next**2 + U_y_next**2)
        
        if U_next_norm < self.U_max:
            #U_x_next, U_y_next = self.BC(x_dog, y_dog, U_x_next, U_y_next)
            output_x = x_dog + U_x_next
            output_y = y_dog + U_y_next
        else:
            U_x_e, U_y_e = self.vector_normalize(U_x_next, U_y_next)
            U_x_next = self.U_max * U_x_e
            U_y_next = self.U_max * U_y_e
            #U_x_next, U_y_next = self.BC(x_dog, y_dog, U_x_next, U_y_next)
            output_x = x_dog + U_x_next
            output_y = y_dog + U_y_next
        return output_x, output_y, U_x_next, U_y_next
    
    
    def step2(self, x_dog, y_dog, x_sheep, y_sheep):
        dx_neighbor, dy_neighbor, dist_neighbor = self.calculate_neighbor_sheep(x_dog, y_dog, x_sheep, y_sheep)
        dist_neighbor_min = np.min(dist_neighbor)
        
        const = np.min(np.array([5, dist_neighbor_min - 3 - 0.001]))
        dx, dy = self.f2_convex(x_dog, y_dog, x_sheep, y_sheep)
        dist = np.sqrt(dx*2 + dy**2)
        dx_normal, dy_normal = self.vector_normalize(dx,dy)
        
        output_x = const * dx_normal if dist > 0 else 0.
        output_y = const * dy_normal if dist > 0 else 0.
        
        x_dog_next = x_dog + output_x
        y_dog_next = y_dog + output_y
        return x_dog_next, y_dog_next
    

#%%
# 牧羊犬の制御モデルの実装
import numpy as np
import random

class Sheepdog_sueoka:
    def __init__(self, num_sheep, k_c1, k_c2, k_c3, k_f1, k_f2, k_f3, k_o, c_0, r_s, r_fn, l_c, l_d, x_target, y_target, condition):
        self.N_s = num_sheep
        self.k_c1 = k_c1
        self.k_c2 = k_c2
        self.k_c3 = k_c3
        self.k_f1 = k_f1
        self.k_f2 = k_f2
        self.k_f3 = k_f3
        self.k_o = k_o
        self.c_0 = c_0
        self.r_s = r_s
        self.r_fn = r_fn
        self.l_c = l_c
        self.l_d = l_d
        self.x_target = x_target
        self.y_target = y_target
        self.condition = condition
        
    def vector_normalize(self, x, y):
        xy = np.column_stack((x, y))
        xy_norm = np.linalg.norm(xy, axis=1)
        # Avoid division by zero by checking if the norm is non-zero
        xy_normalize = np.divide(xy, xy_norm[:, None], out=np.zeros_like(xy, dtype=float), where=(xy_norm[:, None] != 0))
        return xy_normalize[:, 0], xy_normalize[:, 1]
        
    def f1(self, x_d, y_d, x_s, y_s):
        x_centroid = np.nanmean(x_s)
        y_centroid = np.nanmean(y_s)
        
        dx = x_d - x_centroid
        dy = y_d - y_centroid
        dx_normal, dy_normal = self.vector_normalize(dx, dy)
        
        output_x = -dx_normal
        output_y = -dy_normal
        return output_x, output_y
    
    def f2(self, x_d, y_d, x_s, y_s):
        x_centroid = np.nanmean(x_s)
        y_centroid = np.nanmean(y_s)
        
        dx = x_d - x_centroid
        dy = y_d - y_centroid
        dx_norm = np.abs(dx)
        dy_norm = np.abs(dy)
        
        output_x = dx / dx_norm**3 if dx_norm != 0 else 0
        output_y = dy / dy_norm**3 if dy_norm != 0 else 0
        return output_x, output_y
    
    def f3(self, x_d, y_d):
        dx = x_d - self.x_target
        dy = y_d - self.y_target
        
        dx_normal, dy_normal = self.vector_normalize(dx, dy)
        
        output_x = dx_normal
        output_y = dy_normal
        return output_x, output_y
    
    def f4(self, x_d, y_d, x_s, y_s):
        x_s_flockg_c = np.nanmean(x_s)
        y_s_flockg_c = np.nanmean(y_s)
        
        dx_flock_sheep = x_s - x_s_flockg_c
        dy_flock_sheep = y_s - y_s_flockg_c
        dist_flock_sheep = np.sqrt(dx_flock_sheep**2 + dy_flock_sheep**2)
        most_far_sheep_index = np.nanargmax(dist_flock_sheep)
        d_f = np.nanmax(dist_flock_sheep)
        x_s_far_from_flock = x_s[most_far_sheep_index]
        y_s_far_from_flock = y_s[most_far_sheep_index]
        
        # 1
        dx_c = x_s_far_from_flock - x_s_flockg_c
        dy_c = y_s_far_from_flock - y_s_flockg_c
        norm_c = np.sqrt(dx_c**2 + dy_c**2)
        x_p_c = x_s_flockg_c + (self.l_c / norm_c) * dx_c
        y_p_c = y_s_flockg_c + (self.l_c / norm_c) * dy_c
        dx_dog_p_c = x_p_c - x_d # 1
        dy_dog_p_c = y_p_c - y_d # 1
        dx_dog_p_c_normal, dy_dog_p_c_normal = self.vector_normalize(dx_dog_p_c, dy_dog_p_c)
        
        # 2
        dx_c_t = x_s_flockg_c - self.x_target
        dy_c_t = y_s_flockg_c - self.y_target
        norm_c_t = np.sqrt(dx_c_t**2 + dy_c_t**2)
        m = norm_c_t + self.l_d
        n = self.l_d
        x_p_d = (-n * x_s_flockg_c + m * self.x_target) / (m - n)
        y_p_d = (-n * y_s_flockg_c + m * self.y_target) / (m - n)
        dx_dog_p_d = x_p_d - x_d
        dy_dog_p_d = y_p_d - y_d
        dx_dog_p_d_normal, dy_dog_p_d_normal = self.vector_normalize(dx_dog_p_d, dy_dog_p_d)
        
        dist_dog_sheep_min = np.nanmin(np.sqrt((x_d - x_s)**2 + (y_d - y_s)**2))
        
        if d_f > self.r_fn:
            output_x = self.k_o * (-dx_dog_p_c_normal)
            output_y = self.k_o * (-dy_dog_p_c_normal)
        elif d_f <= self.r_fn:
            output_x = self.k_o * (-dx_dog_p_d_normal)
            output_y = self.k_o * (-dy_dog_p_d_normal)
        elif dist_dog_sheep_min < self.c_0 * self.r_s:
            output_x = 0
            output_y = 0
        return output_x, output_y
    
    def f5(self, x_d, y_d, x_s, y_s):
        dist_dog_target = np.sqrt((x_s - self.x_target)**2 + (y_s - self.y_target)**2)
        most_far_sheep_index = np.nanargmax(dist_dog_target)
        x_most_far = x_s[most_far_sheep_index]
        y_most_far = y_s[most_far_sheep_index]
        dx = x_d - x_most_far
        dy = y_d - y_most_far
        dx_normal, dy_normal = self.vector_normalize(dx, dy)
        
        u1_x, u1_y = -dx_normal, -dy_normal
        u2_x, u2_y = dx_normal, dy_normal
        u3_x, u3_y = self.vector_normalize(x_d - self.x_target, y_d - self.y_target)
        
        output_x = self.k_f1 * u1_x + self.k_f2 * u2_x + self.k_f3 * u3_x
        output_y = self.k_f1 * u1_y + self.k_f2 * u2_y + self.k_f3 * u3_y
        return output_x, output_y
    
    def reset(self):
        x = 15
        y = 5
        return x, y
        
    def step(self, x_d, y_d, x_s, y_s):
        if self.condition == 1:
            u1_x, u1_y = self.f1(x_d, y_d, x_s, y_s)
            u2_x, u2_y = self.f2(x_d, y_d, x_s, y_s)
            u3_x, u3_y = self.f3(x_d, y_d)
            u_x = self.k_c1 * u1_x + self.k_c2 * u2_x + self.k_c3 * u3_x
            u_y = self.k_c1 * u1_y + self.k_c2 * u2_y + self.k_c3 * u3_y
        elif self.condition == 2:
            u_x, u_y = self.f4(x_d, y_d, x_s, y_s)
        elif self.condition == 3:
            u_x, u_y = self.f5(x_d, y_d, x_s, y_s)
            
        output_x = x_d + u_x
        output_y = y_d + u_y
        return output_x, output_y


#%%
# 牧羊犬の制御モデル

import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import Polygon
from scipy.spatial import ConvexHull
from sympy import*
import random

class Sheepdog_azuma:
    def __init__(self, num_sheep, x_goal, y_goal, lx, ly):
        self.N = num_sheep
        self.v_d = 10.0
        self.v_s = 4.0
        self.epsilon = 0.001
        self.iota = 5.0
        self.x_goal = x_goal
        self.y_goal = y_goal
        self.lx = lx
        self.ly = ly
        
    def vector_normalize(self, x, y):
        xy = np.column_stack((x, y))
        xy_norm = np.linalg.norm(xy, axis=1)
        # Avoid division by zero by checking if the norm is non-zero
        xy_normalize = np.divide(xy, xy_norm[:, None], out=np.zeros_like(xy, dtype=float), where=(xy_norm[:, None] != 0))
        return xy_normalize[:, 0], xy_normalize[:, 1]
    
    # 凸包の頂点を計算
    def convex(self, x, y):
        index_notnan = np.where(~np.isnan(x))[0]
        x_notnan = x[index_notnan]
        y_notnan = y[index_notnan]
        points = np.column_stack((x_notnan, y_notnan))
        hull = ConvexHull(points)
        points = hull.points
        hull_points = points[hull.vertices]
        return hull_points.T[0], hull_points.T[1]
    
    # ゴールと牧羊犬の位置を結ぶ直線上に羊の座標を射影する関数
    def project_points_onto_line(self, x_d, y_d, px, py):
        dx_x_goal = self.x_goal - x_d
        dy_y_goal = self.y_goal - y_d
        dx_x_goal_normal, dy_y_goal_normal = self.vector_normalize(dx_x_goal, dy_y_goal)
        
        dx_x_px = px - x_d
        dy_y_py = py - y_d
        
        naiseki = (dx_x_px * dx_x_goal_normal) + (dy_y_py * dy_y_goal_normal)
        
        projected_x = naiseki * dx_x_goal_normal + x_d
        projected_y = naiseki * dy_y_goal_normal + y_d
        return projected_x, projected_y
    
    # 動かす羊を選定
    def calculate_C(self, x_dog, y_dog, x_sheep, y_sheep):
        hull_points_x, hull_points_y = self.convex(x_sheep, y_sheep)
        projection_x, projection_y = self.project_points_onto_line(x_dog, y_dog, hull_points_x, hull_points_y)
        
        dx = projection_x - x_dog
        dy = projection_y - y_dog
        dist = np.sqrt(dx**2 + dy**2)
        dist_min_index = np.argmin(dist)
        x_star = hull_points_x[dist_min_index]
        y_star = hull_points_y[dist_min_index]
        return x_star, y_star
    
    def reset(self):
        x = self.lx - 10
        y = self.ly - 10
        return x, y
    
    def step(self, x_dog, y_dog, x_sheep, y_sheep):
        x_star, y_star = self.calculate_C(x_dog, y_dog, x_sheep, y_sheep)
        
        dx_d_dog = (x_dog - x_star) - (x_star - x_star)
        dy_d_dog = (y_dog - y_star) - (y_star - y_star)
        
        dx_star_d1 = np.cos(np.pi/2) * dx_d_dog + (-np.sin(np.pi/2)) * dy_d_dog
        dy_star_d1 = np.sin(np.pi/2) * dx_d_dog + np.cos(np.pi/2) * dy_d_dog
        dx_star_d1_normal, dy_star_d1_normal = self.vector_normalize(dx_star_d1, dy_star_d1)
        dx_star_d2 = np.cos(-np.pi/2) * dx_d_dog + (-np.sin(-np.pi/2)) * dy_d_dog
        dy_star_d2 = np.sin(-np.pi/2) * dx_d_dog + np.cos(-np.pi/2) * dy_d_dog
        dx_star_d2_normal, dy_star_d2_normal = self.vector_normalize(dx_star_d2, dy_star_d2)
        
        x_d1 = self.iota * dx_star_d1_normal + x_star
        y_d1 = self.iota * dy_star_d1_normal + y_star
        x_d2 = self.iota * dx_star_d2_normal + x_star
        y_d2 = self.iota * dy_star_d2_normal + y_star
        
        dist_d_goal1 = np.sqrt((x_d1 - self.x_goal)**2 + (y_d1 - self.y_goal)**2)
        dist_d_goal2 = np.sqrt((x_d2 - self.x_goal)**2 + (y_d2 - self.y_goal)**2)
        x_d = x_d1 if dist_d_goal1 > dist_d_goal2 else x_d2
        y_d = y_d1 if dist_d_goal1 > dist_d_goal2 else y_d2
        
        dx_d_dog_normal, dy_d_dog_normal = self.vector_normalize(x_d - x_dog, y_d - y_dog)
        dist_d_dog = np.sqrt((x_d - x_dog)**2 + (y_d - y_dog)**2)
        dist_d_sheep = np.sqrt((x_d - x_sheep)**2 + (y_d - y_sheep)**2)
        
        v_x = np.nanmin(np.array([self.v_d, np.nanmin(dist_d_sheep) - self.v_s - self.epsilon])) * dx_d_dog_normal if dist_d_dog > 0 else 0
        v_y = np.nanmin(np.array([self.v_d, np.nanmin(dist_d_sheep) - self.v_s - self.epsilon])) * dy_d_dog_normal if dist_d_dog > 0 else 0
        
        x_dog_new = x_dog + 5.0 * v_x
        y_dog_new = y_dog + 5.0 * v_y
        return x_dog_new, y_dog_new
    
    
#%%
# 牧羊犬制御モデル

class Sheepdog_sumida:
    def __init__(self, num_sheep, x_goal, y_goal, lx, ly):
        self.N_s = num_sheep
        self.x_goal = x_goal
        self.y_goal = y_goal
        self.lx = lx
        self.ly = ly
        self.K4 = 8.
        self.K5 = 71.5
        self.K6 = 3.04
        
    def vector_normalize(self, x, y):
        xy = np.column_stack((x, y))
        xy_norm = np.linalg.norm(xy, axis=1)
        # Avoid division by zero by checking if the norm is non-zero
        xy_normalize = np.divide(xy, xy_norm[:, None], out=np.zeros_like(xy, dtype=float), where=(xy_norm[:, None] != 0))
        return xy_normalize[:, 0], xy_normalize[:, 1]
        
    def cul_gravity(self, x_s, y_s):
        x_g = np.nanmean(x_s)
        y_g = np.nanmean(y_s)
        return x_g, y_g
    
    def u_d(self, x_d, y_d, x_s, y_s):
        x_g, y_g = self.cul_gravity(x_s, y_s)
        
        # the first term
        dx1 = x_d - x_g
        dy1 = y_d - y_g
        dx1_normal, dy1_normal = self.vector_normalize(dx1, dy1)
        output_x1, output_y1 = -self.K4 * dx1_normal, -self.K4 * dy1_normal
        
        # the second term
        norm = np.sqrt(dx1**2 + dy1**2)
        dx2 = dx1_normal / norm**2
        dy2 = dy1_normal / norm**2
        output_x2, output_y2 = self.K5 * dx2, self.K5 * dy2
        
        # the third term
        dx3 = x_d - self.x_goal
        dy3 = y_d - self.y_goal
        dx3_normal, dy3_normal = self.vector_normalize(dx3, dy3)
        output_x3, output_y3 = self.K6 * dx3_normal, self.K6 * dy3_normal
        
        # output
        output_x = output_x1 + output_x2 + output_x3
        output_y = output_y1 + output_y2 + output_y3
        return output_x, output_y
    
    def reset(self):
        x = self.lx - 10
        y = self.ly - 10
        return x, y
    
    def step(self, x_d, y_d, x_s, y_s):
        input_x, input_y = self.u_d(x_d, y_d, x_s, y_s)
        
        x_d_next = x_d + input_x
        y_d_next = y_d + input_y
        return x_d_next, y_d_next


#%%
# シミュレーションの実行関数を実装

import numpy as np
import random

# 諸定数
num_sheep = 50
d_ss1 = 2. # 犬がいる
d_ss2 = 5. # 犬がいない
d_gr = 30
d_dg = 50
d_fn = 15
d_dr = 10
d_cl = 5
d_sd = 2
V_max1 = 3. # 牧羊犬がいる
V_max2 = 0.5 # 牧羊犬がいない
U_max = 6.
w0 = 0.5
w1 = 2.
w2_1 = 1.05 # 牧羊犬がいる
w2_2 = 0.01
w3_1 = 1 # 牧羊犬がいる
w3_2 = 0
z0 = 1.5
z1 = 1.5
z2 = 3.
lx = 200
ly = 200
x_start = lx/2
y_start = ly/2
x_target = ly/2
y_target = lx/2
target_area_r = 15
Tcondition = 20
Tmax = 1000

r_s = 20
r_fn = 24.1
k_s1 = 10
k_s2 = 0.5
k_s3 = 2.0
k_s4 = 5.0
k_o = 3.0
k_c1 = k_f1 = 10.0
k_c2 = k_f2 = 200
k_c3 = k_f3 = 8.0
c_0 = 0.001
l_c = 15
l_d = 30
condition = 3

sheep = Sheep(num_sheep, d_ss1, d_ss2, d_gr, d_dg, V_max1, V_max2, w0, w1, w2_1, w2_2, w3_1, w3_2, lx, ly)
#dog = Sheepdog(num_sheep, d_sd, d_gr, d_fn, d_dr, d_cl, U_max, z0, z1, z2, x_start, y_start, x_target, y_target, lx, ly)
#dog_sueoka = Sheepdog_sueoka(num_sheep, k_c1, k_c2, k_c3, k_f1, k_f2, k_f3, k_o, c_0, r_s, r_fn, l_c, l_d, x_target, y_target, condition)
#dog_azuma = Sheepdog_azuma(num_sheep, x_target, y_target, lx, ly)
dog_sumida = Sheepdog_sumida(num_sheep, x_target, y_target, lx, ly)

def simulate(sheep, dog_sumida, Tmax):
    x_sheep, y_sheep, v_x, v_y = sheep.reset()
    #x_dog, y_dog, u_x, u_y = dog.reset()
    #x_dog, y_dog = dog_sueoka.reset()
    #x_dog, y_dog = dog_azuma.reset()
    x_dog, y_dog = dog_sumida.reset()
        
    x_sheep_list = []
    y_sheep_list = []
    x_dog_list = []
    y_dog_list = []
    
    for t in range(Tmax):
        x_sheep_list.append(x_sheep)
        y_sheep_list.append(y_sheep)
        x_dog_list.append(x_dog)
        y_dog_list.append(y_dog)
        
        x_sheep_next, y_sheep_next, v_x_next, v_y_next = sheep.step(x_sheep, y_sheep, x_dog, y_dog, v_x, v_y)
        #x_dog_next, y_dog_next, u_x_next, u_y_next = dog.step(x_dog, y_dog, x_sheep, y_sheep, u_x, u_y)
        #x_dog_next, y_dog_next = dog_sueoka.step(x_dog, y_dog, x_sheep, y_sheep)
        #x_dog_next, y_dog_next = dog_azuma.step(x_dog, y_dog, x_sheep, y_sheep)
        x_dog_next, y_dog_next = dog_sumida.step(x_dog, y_dog, x_sheep, y_sheep)
        
        x_sheep, y_sheep, v_x, v_y = x_sheep_next, y_sheep_next, v_x_next, v_y_next
        #x_dog, y_dog, u_x, u_y = x_dog_next, y_dog_next, u_x_next, u_y_next
        x_dog, y_dog = x_dog_next, y_dog_next
        
        progress_rate = (t+1)/Tmax*100
        print('Progress={:.1f}%'.format(progress_rate), end='\r')
        
    return x_sheep_list, y_sheep_list, x_dog_list, y_dog_list

# 囲い込んだ羊は消滅する
def condition_dead_or_alive(x, y):
    dx = x_target - x
    dy = y_target - y
    condition = np.sqrt(dx**2 + dy**2) < target_area_r
    return condition

def flag(x1, x2, x3, x4, condition):
    output_x1 = np.where(condition==True, np.nan, x1)
    output_x2 = np.where(condition==True, np.nan, x2)
    output_x3 = np.where(condition==True, np.nan, x3)
    output_x4 = np.where(condition==True, np.nan, x4)
    return output_x1, output_x2, output_x3, output_x4

def simulate2(sheep, dog_azuma, Tmax):
    x_sheep, y_sheep, v_x, v_y = sheep.reset()
    #x_dog, y_dog, u_x, u_y = dog.reset()
    #x_dog, y_dog = dog_sueoka.reset()
    x_dog, y_dog = dog_azuma.reset()
    
    x_sheep_list = []
    y_sheep_list = []
    x_dog_list = []
    y_dog_list = []
    
    conditions = np.zeros(num_sheep)
    
    for t in range(Tmax):
        x_sheep_list.append(x_sheep)
        y_sheep_list.append(y_sheep)
        x_dog_list.append(x_dog)
        y_dog_list.append(y_dog)
        
        condition_index = condition_dead_or_alive(x_sheep, y_sheep)
        conditions[condition_index] += 1
        condition_t = conditions > Tcondition
        
        if np.sum(condition_t) == num_sheep:
            Tbreak = t
            print('Complete! Break time={}'.format(t))
            break
        else:
            Tbreak = Tmax
        
        x_sheep_next, y_sheep_next, v_x_next, v_y_next = sheep.step(x_sheep, y_sheep, x_dog, y_dog, v_x, v_y)
        #x_dog_next, y_dog_next, u_x_next, u_y_next = dog.step(x_dog, y_dog, x_sheep, y_sheep, u_x, u_y)
        #x_dog_next, y_dog_next = dog_sueoka.step(x_dog, y_dog, x_sheep, y_sheep)
        x_dog_next, y_dog_next = dog_azuma.step(x_dog, y_dog, x_sheep, y_sheep)
        
        x_sheep, y_sheep, v_x, v_y = flag(x_sheep_next, y_sheep_next, v_x_next, v_y_next, condition_t)
        #x_dog, y_dog, u_x, u_y = x_dog_next, y_dog_next, u_x_next, u_y_next
        x_dog, y_dog = x_dog_next, y_dog_next
        
        progress_rate = (t+1)/Tmax*100
        print('Progress={:.1f}%'.format(progress_rate), end='\r')
        
    return x_sheep_list, y_sheep_list, x_dog_list, y_dog_list, Tbreak
        

#%%
# シミュレーションを実行

import numpy as np

x_sheep, y_sheep, x_dog, y_dog = simulate(sheep, dog_sumida, Tmax)
#x_sheep, y_sheep, x_dog, y_dog, T_break = simulate2(sheep, dog_azuma, Tmax)
        
        
#%%
# アニメーションを作成
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches

#plt.scatter(x_sheep[0], y_sheep[0])

# 保存先のファイルを作成
folder_name = "Sheepdog"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# キャンバスを設置
fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot()

ax1.set_xlim(0, lx)
ax1.set_ylim(0, ly)

#アップデート関数
def update1(i):
    data_x_dog = x_dog[i]
    data_y_dog = y_dog[i]
    data_x_sheep = x_sheep[i][~np.isnan(x_sheep[i])]
    data_y_sheep = y_sheep[i][~np.isnan(y_sheep[i])]
    nan_count = np.sum(np.isnan(x_sheep[i]))
        
    ax1.clear()
        
    ax1.set_xlim(0, lx)
    ax1.set_ylim(0, ly)
        
    #各タイトル
    #ax1.set_title('N={}, Tmax={}, t={}, Tbreak={}, target=({}, {}), Enclosed sheep={}'.format(num_sheep, Tmax, i+1, T_break, x_target, y_target, nan_count), fontsize=14)
    ax1.set_title('N={}, Tmax={}, t={}, target=({}, {})'.format(num_sheep, Tmax, i+1, x_target, y_target), fontsize=14)
    
    
    #c2 = patches.Circle(xy=(lx/2, ly/2), radius=lx/2, fc='White', ec='Orange', alpha=1.)
    #ax1.add_patch(c2)
    
    #エージェントベクトル更新
    #points = np.column_stack((data_x_sheep, data_y_sheep))
    #hull = ConvexHull(points)
    #points = hull.points
    #hull_points = points[hull.vertices]
    
    #hp = np.vstack((hull_points, hull_points[0]))
    #ax1.plot(hp[:,0], hp[:,1], color='blue', alpha=0.5)
    #ax1.scatter(points[:,0], points[:,1], color='blue')
    
    ax1.scatter(data_x_sheep, data_y_sheep, color='blue')
    ax1.scatter(data_x_dog, data_y_dog, color='red')
    ax1.plot(x_target, y_target, marker='s', color='g')
    
    c1 = patches.Circle(xy=(x_target, y_target), radius=target_area_r, fc='g', alpha=0.4)
    ax1.add_patch(c1)
        
    # 進捗状況を出力する
    progress_rate = (i+1)/Tmax*100
    print("Animation progress={:.3f}%".format(progress_rate), end='\r')
        
#アニメーション作成とgif保存
ani = animation.FuncAnimation(fig, update1, frames=range(Tmax))
    
#グラフ表示
plt.show()

#アニメーションの保存
#output_file = os.path.join(folder_name, 'SheepDog_sumida1.gif')
#ani.save(output_file, writer='pillow')


# 強化学習を用いた協調性について
#%%
# Qネットの実装

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
        
        #if next_q.size() != reward.size():
            #print('reward={}'.format(reward.size()))
            #print('next_q={}'.format(next_q.size()))
        
        target = reward + (1 - done) + self.gamma * next_q #教師信号を算出
        
        criterion = nn.MSELoss()
        loss = criterion(q, target) #最小二乗誤差を計算
            
        self.optimizer.zero_grad() #勾配の初期化
        loss.backward() #誤差を逆伝播
        self.optimizer.step() #重みの更新


#%%
# 学習環境の実装

class Agent_dog:
    def __init__(self, num_dog, num_sheep, lx, ly):
        self.N_d = num_dog
        self.N_s = num_sheep
        self.lx = lx
        self.ly = ly
        self.x_goal = lx/2
        self.y_goal = ly/2
        
    def reset(self):
        x = np.full(self.N_d, 0)
        y = np.full(self.N_d, ly/2)
        return x, y
    
    def get_reward(self, x_d, y_d, size_past, size_next):
        # 領域内にいる報酬
        flag_x = np.where(0 < x_d < self.lx, 1, 0)
        flag_y = np.where(0 < y_d < self.ly, 1, 0)
        reward_area = np.where(flag_x * flag_y == 1, 1, -1)
        
        flag_size = np.where(size_next < size_past, 1, -1)
        reward_enclose = np.full(self.N_d, flag_size)
        
        reward = reward_area + reward_enclose
        return reward
    
    def vector_normalize(self, x, y):
        xy = np.column_stack((x, y))
        xy_norm = np.linalg.norm(xy, axis=1)
        # Avoid division by zero by checking if the norm is non-zero
        xy_normalize = np.divide(xy, xy_norm[:, None], out=np.zeros_like(xy, dtype=float), where=(xy_norm[:, None] != 0))
        return xy_normalize[:, 0], xy_normalize[:, 1]
    
    # 凸包の頂点を計算
    def convex(self, x, y):
        index_notnan = np.where(~np.isnan(x))[0]
        x_notnan = x[index_notnan]
        y_notnan = y[index_notnan]
        points = np.column_stack((x_notnan, y_notnan))
        hull = ConvexHull(points)
        points = hull.points
        hull_points = points[hull.vertices]
        return hull_points.T[0], hull_points.T[1]
    
    # ゴールと牧羊犬の位置を結ぶ直線上に羊の座標を射影する関数
    def project_points_onto_line(self, x_d, y_d, px, py):
        dx_x_goal = self.x_goal - x_d
        dy_y_goal = self.y_goal - y_d
        dx_x_goal_normal, dy_y_goal_normal = self.vector_normalize(dx_x_goal, dy_y_goal)
        
        dx_x_px = px - x_d
        dy_y_py = py - y_d
        
        naiseki = (dx_x_px * dx_x_goal_normal) + (dy_y_py * dy_y_goal_normal)
        
        projected_x = naiseki * dx_x_goal_normal + x_d
        projected_y = naiseki * dy_y_goal_normal + y_d
        return projected_x, projected_y
    
    # 動かす羊を選定
    def calculate_C(self, x_dog, y_dog, x_sheep, y_sheep):
        hull_points_x, hull_points_y = self.convex(x_sheep, y_sheep)
        projection_x, projection_y = self.project_points_onto_line(x_dog, y_dog, hull_points_x, hull_points_y)
        
        dx = projection_x - x_dog
        dy = projection_y - y_dog
        dist = np.sqrt(dx**2 + dy**2)
        dist_min_index = np.argmin(dist)
        x_star = hull_points_x[dist_min_index]
        y_star = hull_points_y[dist_min_index]
        return x_star, y_star
    
    def get_state(self, x_d, y_d, x_s, y_s):
        x_star, y_star = self.calculate_C(x_d, y_d, x_s, y_s)
        
        state = np.array([x_star, y_star, x_d[0], y_d[0]])
        #print('state={}'.format(state))
        return state
    
    def step(self, x_d, y_d, action_index):
        actions = np.array([[5,0], [0,5], [-5,0], [0,-5], [0,0]])
        action = actions[action_index]
        #print('index={}, action={}, type={}'.format(action_index, action, type(action)))
        
        x_d_next = x_d + action[0]
        y_d_next = y_d + action[1]
        return x_d_next, y_d_next
    
    
#%%
# 学習実行
import numpy as np
import os
import torch
import time

class Learn:
    def __init__(self, num_dog, num_sheep, lx, ly):
        self.N_d = num_dog
        self.N_s = num_sheep
        self.lx = lx
        self.ly = ly
        
        self.action_size = 5
        self.state_size = 4
        self.policy = 'epsilon_greedy'
        self.Tmax = 2000
        self.episodes = 10000
        self.sync_interval = 20
        
        d_ss1 = 2. # 犬がいる
        d_ss2 = 5. # 犬がいない
        d_gr = 30
        d_dg = 50
        V_max1 = 3. # 牧羊犬がいる
        V_max2 = 0.5 # 牧羊犬がいない
        w0 = 0.5
        w1 = 2.
        w2_1 = 1.05 # 牧羊犬がいる
        w2_2 = 0.01
        w3_1 = 1 # 牧羊犬がいる
        w3_2 = 0
        
        self.agent = dqn_agent(self.action_size, self.state_size, self.policy, self.N_d)
        self.dog = Agent_dog(self.N_d, self.N_s, self.lx, self.ly)
        self.sheep = Sheep(self.N_s, d_ss1, d_ss2, d_gr, d_dg, V_max1, V_max2, w0, w1, w2_1, w2_2, w3_1, w3_2, self.lx, self.ly)
    
    def save(self, tag):
        save_dir = 'my_rl_Sheepdog/'

        # ディレクトリが存在しない場合は作成する
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 1. モデルの保存ディレクトリを指定する
        save_dir = 'my_rl_Sheepdog/'

        # 2. モデルの重みを取得する
        model = self.agent.qnet.state_dict()
            
        # 4. ファイル名の指定
        file_name = 'Sheepdog' + tag + '.pth'

        # 3. 重みを保存する
        torch.save(model, save_dir + file_name)
        print('File name is {}'.format(file_name))
            
    def cal_size(self, x_s, y_s):
        x_g = np.nanmean(x_s)
        y_g = np.nanmean(y_s)
        dist = np.sqrt((x_s - x_g)**2 + (y_s - y_g)**2)
        size = np.nanmax(dist)
        return size
        
    def run(self, tag):
        reward_episodes = []
        size_episodes = []
        for ep in range(self.episodes):
            tortal_reward = 0
            tortal_size = 0
            counter = 0
            
            x_d, y_d = self.dog.reset()
            x_s, y_s, v_x, v_y = self.sheep.reset()
            
            for t in range(self.Tmax):
                state = self.dog.get_state(x_d, y_d, x_s, y_s)
                action = self.agent.get_action(state)
                
                x_d_next, y_d_next = self.dog.step(x_d, y_d, action)
                x_s_next, y_s_next, v_x_next, v_y_next = self.sheep.step(x_s, y_s, x_d, y_d, v_x, v_y)
                state_next = self.dog.get_state(x_d_next, y_d_next, x_s_next, y_s_next)
                
                size_past = self.cal_size(x_s, y_s)
                size_next = self.cal_size(x_s_next, y_s_next)
                if t > self.Tmax * 0.8:
                    tortal_size += size_next
                    counter += 1
                reward = self.dog.get_reward(x_d, y_d, size_past, size_next)
                
                self.agent.update(state, action, reward, state_next, done=False)
                
                x_d, y_d = x_d_next, y_d_next
                x_s, y_s, v_x, v_y = x_s_next, y_s_next, v_x_next, v_y_next
                
                tortal_reward += float(reward)
                
                if t % (self.Tmax / 100) == 0:
                    print("progress={}%".format(int(t/self.Tmax * 100)), end="\r")
                    time.sleep(0.1)
                
            if ep % self.sync_interval == 0:
                self.agent.sync_qnet()
                
            convergence_size = tortal_size / counter
            
            print('Ep={}, Reward={:.3f}, convergence value of size={:.3f}'.format(ep+1, tortal_reward, convergence_size))
            
            reward_episodes.append(tortal_reward)
            size_episodes.append(convergence_size)
            
        self.save(tag)
        
        return reward_episodes, size_episodes


#%%
# 学習を実行する
import numpy as np

num_dog = 1
num_sheep = 50
lx = 200
ly = 200

learning_machine = Learn(num_dog, num_sheep, lx, ly)

tag = 1
rewards, sizes = learning_machine.run(tag)


#%%
#データの書き出し
import os
import numpy as np

folder_name = "sheepdog_rl_folder"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

file_path = os.path.join(folder_name, "data_reward_sheepdog_rl1.csv")
data_rewards = np.array(rewards)
np.savetxt(file_path, data_rewards)

file_path = os.path.join(folder_name, "data_sizes_sheepdog_rl1.csv")
data_file_names = np.array(sizes)
np.savetxt(file_path, data_file_names, fmt='%s')
# %%
