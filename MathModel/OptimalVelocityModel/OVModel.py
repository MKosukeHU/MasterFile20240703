#%%
import numpy as np
import random

class OVmodel:
    def __init__(self, num, alpha, beta, a, c, d, lx, ly):
        self.N = num
        self.alpha = alpha
        self.beta = beta
        self.a = a
        self.c = c
        self.d = d
        self.lx = lx
        self.ly = ly


    def F(self, x1, x2, z1, z2):
        dx1 = x1[np.newaxis,:] - x1[:,np.newaxis]
        dx2 = x2[np.newaxis,:] - x2[:,np.newaxis]
        for i in range(self.N):
            dx1_ij = dx1[i]
            dx2_ij = dx2[i]
            flag_abs_x1 = np.where(np.abs(dx1) <= self.lx/2, 0, 1)
            flag_abs_x2 = np.where(np.abs(dx2) <= self.ly/2, 0, 1)
            flag_pos_x1 = np.where(dx1_ij >= 0, 1, 2)
            flag_pos_x2 = np.where(dx2_ij >= 0, 1, 2)
            flag_x1 = flag_abs_x1 * flag_pos_x1
            flag_x2 = flag_abs_x2 * flag_pos_x2

            dx1[i] = np.where(flag_x1 == 1, dx1_ij - self.lx , np.where(flag_x1 == 2, dx1_ij + self.lx, dx1_ij))
            dx1[i] = np.where(flag_x2 == 1, dx2_ij - self.ly , np.where(flag_x2 == 2, dx2_ij + self.ly, dx2_ij))
        r = np.sqrt(dx1**2 + dx2**2)
        dx_norm = np.sqrt(dx1**2 + dx2**2)
        e1 = np.where(dx_norm != 0., dx1/dx_norm, dx1)
        e2 = np.where(dx_norm != 0., dx2/dx_norm, dx2)
        
        z_norm = np.sqrt(z1**2 + z2**2)
        cos_theta = (z1 * dx1 + z2 * dx2) / (z_norm * r)
        theta = np.arccos(cos_theta)

        first1 = []
        first2 = []
        for i in range(self.N):
            r_ij = r[i]
            theta_ij = theta[i]
            e1_ij = e1[i]
            e2_ij = e2[i]
            alpha_i = self.alpha[i]
            beta_i = self.beta[i]
            c_i = self.c[i]
            d_i = self.d[i]

            f = alpha_i * (np.tanh(beta_i * (r_ij - d_i) + c_i))
            F1 = f * (1 + np.cos(theta_ij)) * e1_ij
            F2 = f * (1 + np.cos(theta_ij)) * e2_ij

            first1_i = np.sum(F1)
            first2_i = np.sum(F2)
            first1.append(first1_i)
            first2.append(first2_i)
        first1 = np.array(first1)
        first2 = np.array(first2)

        F_x1 = self.a * (first1 + z1)
        F_x2 = self.a * (first2 + z2)
        return F_x1, F_x2
    

    def reset(self):
        x1 = np.random.uniform(low=0, high=self.lx, size=self.N)
        x2 = np.arange.uniform(low=0, high=self.ly, size=self.N)
        z1 = np.zeros(self.N)
        z2 = np.zeros(self.N)
        return x1, x2, z1, z2
    

    def step(self, Tmax, dt):
        x1, x2, z1, z2 = self.reset()

        x1_record = []
        x2_record = []
        z1_record = []
        z2_record = []

        Time = int(Tmax / dt)
        for t in range(Time):
            # 時刻 t における力 F(t) の計算
            F_x1, F_x2 = self.F(x1, x2, z1, z2)

            # 時刻 t + dt における位置 x(t + dt) の計算
            x1_next = x1 + dt * (1 - self.a) * z1 + (dt**2 / 2) * F_x1
            x2_next = x2 + dt * (1 - self.a) * z2 + (dt**2 / 2) * F_x2

            # 時刻 t + dt における速度 z(t + dt) の計算
            z1_next = (2 / (2 + self.a * dt)) * ((2 / (2 - self.a * dt)) * z1 + (dt / 2) * (F_x1 + F_x1_next))
            



#%%
import numpy as np
a = np.arange(25).reshape((5,5))
b = a+1
c = a/b
print(c)

for i in range(5):
    c_i = c[i]
    c[i] = np.where(c_i > 0.9, 0, c_i)

print(c)
# %%
import numpy as np

a = np.arange(5)
b = np.eye(5)
c = a*b
print(c)
# %%
