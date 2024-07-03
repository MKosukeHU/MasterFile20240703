#%%
# Social Force Modelの実装を行う。

import numpy as np

class SFmodel:
    def __init__(self, num, v0, A, B):
        self.N = num
        self.v0 = v0
        self.A = A
        self.B = B

    def f_ij(self, x, y):
        dx = x[np.newaxis,:] - x[:,np.newaxis]
        dy = y[np.newaxis,:] - y[:,np.newaxis]
        d = np.sqrt(dx**2 + dy**2)

        n_x = np.where(d > 0, -dx / d, 0)
        n_y = np.where(d > 0, -dy / d, 0)

        f_ij_x = self.A * np.exp(-d / self.B) * n_x
        f_ij_y = self.A * np.exp(-d / self.B) * n_y

        output_x = np.sum(f_ij_x, axis=1)
        output_y = np.sum(f_ij_y, axis=1)
        return output_x, output_y
    
    def f_iW(self, x, y, walls)
