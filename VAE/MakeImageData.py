"""
目標：
エージェントの座標データから、一定の指標に基づいた画像データの生成プログラムの実装

参考資料：
[1] 
"""


#%%
import numpy as np
import matplotlib.pyplot as plt

# エージェントの位置データを生成（例としてランダムに配置）
np.random.seed(0)
agents = np.random.rand(500, 2) * 10  # 100個のエージェントを[0, 10] x [0, 10]の範囲に配置

# 中心エージェントの座標（例として[5, 5]を中心とする）
center_agent = np.array([5, 5])

# 関心領域の境界を定義
region_size = 1
min_x, max_x = center_agent[0] - region_size / 2, center_agent[0] + region_size / 2
min_y, max_y = center_agent[1] - region_size / 2, center_agent[1] + region_size / 2

# 領域内の隣接エージェントを抽出
neighboring_agents = agents[
    (agents[:, 0] >= min_x) & (agents[:, 0] <= max_x) & 
    (agents[:, 1] >= min_y) & (agents[:, 1] <= max_y)
]

# 可視化
plt.figure(figsize=(8, 8))
plt.scatter(agents[:, 0], agents[:, 1], c='blue', label='Agents')
plt.scatter(center_agent[0], center_agent[1], c='red', label='Center Agent')
plt.scatter(neighboring_agents[:, 0], neighboring_agents[:, 1], c='green', label='Neighboring Agents')
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.axvline(x=min_x, color='gray', linestyle='--')
plt.axvline(x=max_x, color='gray', linestyle='--')
plt.axhline(y=min_y, color='gray', linestyle='--')
plt.axhline(y=max_y, color='gray', linestyle='--')
plt.legend()
plt.title('Localized Distribution of Neighboring Agents')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.grid(True)
plt.show()


# %%
import numpy as np
import matplotlib.pyplot as plt

# エージェントの位置データを生成（例としてランダムに配置）
np.random.seed(0)
agents = np.random.rand(1000000, 2) * 7  # 1000個のエージェントを[0, 10] x [0, 10]の範囲に配置

# 中心エージェントの座標（例として[5, 5]を中心とする）
center_agent = np.array([5, 5])

# 関心領域の境界を定義
region_size = 1
min_x, max_x = center_agent[0] - region_size / 2, center_agent[0] + region_size / 2
min_y, max_y = center_agent[1] - region_size / 2, center_agent[1] + region_size / 2

# 領域内の隣接エージェントを抽出
neighboring_agents = agents[
    (agents[:, 0] >= min_x) & (agents[:, 0] <= max_x) & 
    (agents[:, 1] >= min_y) & (agents[:, 1] <= max_y)
]

# グリッドサイズを定義
grid_size = 5

# 各グリッドセルにおけるエージェントの密度を計算
density = np.zeros((grid_size, grid_size))
x_edges = np.linspace(min_x, max_x, grid_size + 1)
y_edges = np.linspace(min_y, max_y, grid_size + 1)

for i in range(grid_size):
    for j in range(grid_size):
        x0, x1 = x_edges[i], x_edges[i+1]
        y0, y1 = y_edges[j], y_edges[j+1]
        count = np.sum((neighboring_agents[:, 0] >= x0) & (neighboring_agents[:, 0] < x1) & 
                       (neighboring_agents[:, 1] >= y0) & (neighboring_agents[:, 1] < y1))
        density[j, i] = count

# 密度を0から1の範囲に正規化
density = density / density.max()

# 可視化
plt.figure(figsize=(8, 8))
plt.imshow(density, extent=[min_x, max_x, min_y, max_y], origin='lower', cmap='hot', interpolation='nearest')
plt.colorbar(label='Density')
plt.scatter(center_agent[0], center_agent[1], c='blue', label='Center Agent', edgecolor='white')
plt.title('Density of Neighboring Agents')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend()
plt.grid(True)
plt.show()


# %%
import numpy as np
import matplotlib.pyplot as plt
import os

def generate_density_images(num_images, agents_per_image, region_size, grid_size, save_dir):
    """
    エージェントの密度画像を生成して保存する関数

    Parameters:
    num_images (int): 生成する画像の数
    agents_per_image (int): 各画像に配置するエージェントの数
    region_size (float): 関心領域のサイズ
    grid_size (int): グリッドサイズ
    save_dir (str): 画像を保存するディレクトリ
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for img_idx in range(num_images):
        # エージェントの位置データを生成
        agents = np.random.rand(agents_per_image, 2) * 7
        
        # 中心エージェントの座標（例として[5, 5]を中心とする）
        center_agent = np.array([5, 5])

        # 関心領域の境界を定義
        min_x, max_x = center_agent[0] - region_size / 2, center_agent[0] + region_size / 2
        min_y, max_y = center_agent[1] - region_size / 2, center_agent[1] + region_size / 2

        # 領域内の隣接エージェントを抽出
        neighboring_agents = agents[
            (agents[:, 0] >= min_x) & (agents[:, 0] <= max_x) & 
            (agents[:, 1] >= min_y) & (agents[:, 1] <= max_y)
        ]

        # 各グリッドセルにおけるエージェントの密度を計算
        density = np.zeros((grid_size, grid_size))
        x_edges = np.linspace(min_x, max_x, grid_size + 1)
        y_edges = np.linspace(min_y, max_y, grid_size + 1)

        for i in range(grid_size):
            for j in range(grid_size):
                x0, x1 = x_edges[i], x_edges[i+1]
                y0, y1 = y_edges[j], y_edges[j+1]
                count = np.sum((neighboring_agents[:, 0] >= x0) & (neighboring_agents[:, 0] < x1) & 
                               (neighboring_agents[:, 1] >= y0) & (neighboring_agents[:, 1] < y1))
                density[j, i] = count

        # 密度を0から1の範囲に正規化
        density = density / density.max()

        # 画像として保存
        plt.figure(figsize=(4, 4))
        plt.imshow(density, extent=[min_x, max_x, min_y, max_y], origin='lower', cmap='hot', interpolation='nearest')
        plt.colorbar(label='Density')
        plt.scatter(center_agent[0], center_agent[1], c='blue', label='Center Agent', edgecolor='white')
        plt.title(f'Density of Neighboring Agents {img_idx + 1}')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, f'density_image_{img_idx + 1}.png'))
        plt.close()

# 使用例
generate_density_images(num_images=100, agents_per_image=1000, region_size=1, grid_size=5, save_dir='vae_training_images')
# %%
