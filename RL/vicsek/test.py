#%%
import numpy as np

num = 10

class Agent:
    def __init__(self):
        self.pack = 'test print'

    def shake():
        print('test')

    def re():
        return 1

agents = [Agent for i in range(num)]

# %%
print(agents[1])
# %%
agent1 = agents[1]
agent1.shake()
cafe = [agents[i].re() for i in range(num)]
print(cafe)
# %%
import numpy as np
a = np.arange(2)
b = a[np.newaxis,:] - a[:,np.newaxis]
c = b < 0
d = np.where(c==1, 10, 9)
e = b[0] * b[1]
print('a={}, b={}, c={}, d={}, e={}'.format(a, b, c, d, e))
# %%
