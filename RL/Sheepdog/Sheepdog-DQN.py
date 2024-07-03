

"""
 ▄▄▄▄▄▄▄▄▄▄   ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄       ▄▄▄▄▄▄▄▄▄▄▄  ▄            ▄▄▄▄▄▄▄▄▄▄▄  ▄▄        ▄ 
▐░░░░░░░░░░▌ ▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌     ▐░░░░░░░░░░░▌▐░▌          ▐░░░░░░░░░░░▌▐░░▌      ▐░▌
▐░█▀▀▀▀▀▀▀█░▌▐░█▀▀▀▀▀▀▀█░▌▐░█▀▀▀▀▀▀▀▀▀  ▀▀▀▀█░█▀▀▀▀ ▐░█▀▀▀▀▀▀▀▀▀      ▐░█▀▀▀▀▀▀▀█░▌▐░▌          ▐░█▀▀▀▀▀▀▀█░▌▐░▌░▌     ▐░▌
▐░▌       ▐░▌▐░▌       ▐░▌▐░▌               ▐░▌     ▐░▌               ▐░▌       ▐░▌▐░▌          ▐░▌       ▐░▌▐░▌▐░▌    ▐░▌
▐░█▄▄▄▄▄▄▄█░▌▐░█▄▄▄▄▄▄▄█░▌▐░█▄▄▄▄▄▄▄▄▄      ▐░▌     ▐░▌               ▐░█▄▄▄▄▄▄▄█░▌▐░▌          ▐░█▄▄▄▄▄▄▄█░▌▐░▌ ▐░▌   ▐░▌
▐░░░░░░░░░░▌ ▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌     ▐░▌     ▐░▌               ▐░░░░░░░░░░░▌▐░▌          ▐░░░░░░░░░░░▌▐░▌  ▐░▌  ▐░▌
▐░█▀▀▀▀▀▀▀█░▌▐░█▀▀▀▀▀▀▀█░▌ ▀▀▀▀▀▀▀▀▀█░▌     ▐░▌     ▐░▌               ▐░█▀▀▀▀▀▀▀▀▀ ▐░▌          ▐░█▀▀▀▀▀▀▀█░▌▐░▌   ▐░▌ ▐░▌
▐░▌       ▐░▌▐░▌       ▐░▌          ▐░▌     ▐░▌     ▐░▌               ▐░▌          ▐░▌          ▐░▌       ▐░▌▐░▌    ▐░▌▐░▌
▐░█▄▄▄▄▄▄▄█░▌▐░▌       ▐░▌ ▄▄▄▄▄▄▄▄▄█░▌ ▄▄▄▄█░█▄▄▄▄ ▐░█▄▄▄▄▄▄▄▄▄      ▐░▌          ▐░█▄▄▄▄▄▄▄▄▄ ▐░▌       ▐░▌▐░▌     ▐░▐░▌
▐░░░░░░░░░░▌ ▐░▌       ▐░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌     ▐░▌          ▐░░░░░░░░░░░▌▐░▌       ▐░▌▐░▌      ▐░░▌
 ▀▀▀▀▀▀▀▀▀▀   ▀         ▀  ▀▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀▀       ▀            ▀▀▀▀▀▀▀▀▀▀▀  ▀         ▀  ▀        ▀▀                                                                                                                                                                        
Font : Electronic
"""

# 2024年度　研究計画概要　最終更新：2024-04-05

# 目的：複数個体での牧羊犬モデルを制御最適化した場合、協調的行動が出現するのかどうか、出現するならばその条件は何であるかをMulti-DQNを用いて調査する。

# 調査方法：Multi-DQNによる複数個体の牧羊犬の制御最適化。羊は参考の数理モデルに従って行動するものとする。

# 昨年の反省：
# ・研究の進捗を随時管理できていなかったので、途中でどこまで何をしたかがわからなくなった。> 研究の進捗は作業ごとにつける。研究進捗欄による管理。
# ・締め切り前になって論文を書き始めたがために、缶詰状態になってしまった。> 研究詳報を頻繁に更新し、ちょくちょく書き進める。研究詳報による管理。
# ・何を参考にしたかが途中でわからなくなった。参考文献管理の怠惰。> 参考文献の管理。参考文献の明示。
# ・研究目的が漠然的すぎたので、途中で迷走しがちだった。> 研究目標の明確化、明文化。

# To do：
# done, 羊の制御モデルの実装。
# done, Single-DQNモデルを実装。
# 2-2, Multi-DQNモデルの実装。（2-1を検証してから進める）
# 3, 環境、観測、行動、報酬を選定、実装。
# 4, シミュレーション、学習の条件を決定。評価基準を決定。
# 5, 学習の実行。
# 6, 性能評価を行う。
# 7, 報告書にまとめる。

# 文献：
# [1]末岡 et al., 複数のロボットによるシープドッグシステムの分散制御, 2014
# [2]東 et al., 牧羊犬制御のモデル化, 2012
# [3]渡辺 & 藤岡, 二匹の牧羊犬を用いた群れ誘導における誘導位置の検証, 2017
# [4]末岡 & 角田, シープドッグシステムから紐解くマルチエージェントシステムの制御, 2020
# [5]末岡 et al., シープドッグシステムにおける誘導制御と性能解析:離散解析によるアプローチ, 2014
# [6]林 & 藤岡, マルチエージェントシステムを利用した効率的な群れ誘導方法の検討, 2016
# [7]筒井 et al., マルチエージェント深層強化学習を用いた協調的かりに見られる行動方略の多様性, 2022
# [8]末岡 et al., 機械学習アプローチから探す群れ行動の発現機序-捕食者からの逃避戦略の学習に基づく自発的な群れ行動の獲得-, 2022
# [9]Tsutsui et al., Synergizing Deep Reinforcement Learning and Biological Pursuit Behavioral Rule for Robust and Interpretable Navigation, 2023
# [10]https://cattech-lab.com/science-tools/sheepdog-simulation/#ref1
# [11]礒川 et al., 強化学習に基づく群行動制御モデルの構成, 2019, https://www.sice.or.jp/wp-content/uploads/file/ci/CI15_all.pdf
# [12]Tashiro et al., Guidance by multiple sheepdogs including abnormalities, 2022
# [13]角田 et al., シープドッグシステムに学ぶエージェント群の機動制御則の設計法と実機検証, 2021

"""
 ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄ 
▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌
▐░█▀▀▀▀▀▀▀█░▌▐░█▀▀▀▀▀▀▀▀▀ ▐░█▀▀▀▀▀▀▀█░▌▐░█▀▀▀▀▀▀▀█░▌▐░█▀▀▀▀▀▀▀█░▌ ▀▀▀▀█░█▀▀▀▀ 
▐░▌       ▐░▌▐░▌          ▐░▌       ▐░▌▐░▌       ▐░▌▐░▌       ▐░▌     ▐░▌     
▐░█▄▄▄▄▄▄▄█░▌▐░█▄▄▄▄▄▄▄▄▄ ▐░█▄▄▄▄▄▄▄█░▌▐░▌       ▐░▌▐░█▄▄▄▄▄▄▄█░▌     ▐░▌     
▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░▌       ▐░▌▐░░░░░░░░░░░▌     ▐░▌     
▐░█▀▀▀▀█░█▀▀ ▐░█▀▀▀▀▀▀▀▀▀ ▐░█▀▀▀▀▀▀▀▀▀ ▐░▌       ▐░▌▐░█▀▀▀▀█░█▀▀      ▐░▌     
▐░▌     ▐░▌  ▐░▌          ▐░▌          ▐░▌       ▐░▌▐░▌     ▐░▌       ▐░▌     
▐░▌      ▐░▌ ▐░█▄▄▄▄▄▄▄▄▄ ▐░▌          ▐░█▄▄▄▄▄▄▄█░▌▐░▌      ▐░▌      ▐░▌     
▐░▌       ▐░▌▐░░░░░░░░░░░▌▐░▌          ▐░░░░░░░░░░░▌▐░▌       ▐░▌     ▐░▌     
 ▀         ▀  ▀▀▀▀▀▀▀▀▀▀▀  ▀            ▀▀▀▀▀▀▀▀▀▀▀  ▀         ▀       ▀         
"""                                                                                            

# 2024年度　研究詳報　最終更新：2024-04-03

# 概要
# 　狩りや移動等における個体間の協調的行動は、微生物から大型哺乳類まで広く観察される。
# 例えば、狩りにおける協調的行動には、単に各個体が自身の利益のために同時に狩っている場合もあれば、互いに補完的な役割を持って狩りを行う場合もある。
# 　これら協調的行動が出現する要因について、[7]では深層強化学習を用いたマルチエージェントシミュレーションを用いた研究を行い、互いが補完的な行動をする狩りと
# 自身の利益のために同時に狩りをする場合との出現要因を発見した。


"""
 ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄ 
▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌
▐░█▀▀▀▀▀▀▀█░▌▐░█▀▀▀▀▀▀▀█░▌▐░█▀▀▀▀▀▀▀█░▌▐░█▀▀▀▀▀▀▀▀▀ ▐░█▀▀▀▀▀▀▀█░▌▐░█▀▀▀▀▀▀▀▀▀ ▐░█▀▀▀▀▀▀▀▀▀ ▐░█▀▀▀▀▀▀▀▀▀ 
▐░▌       ▐░▌▐░▌       ▐░▌▐░▌       ▐░▌▐░▌          ▐░▌       ▐░▌▐░▌          ▐░▌          ▐░▌          
▐░█▄▄▄▄▄▄▄█░▌▐░█▄▄▄▄▄▄▄█░▌▐░▌       ▐░▌▐░▌ ▄▄▄▄▄▄▄▄ ▐░█▄▄▄▄▄▄▄█░▌▐░█▄▄▄▄▄▄▄▄▄ ▐░█▄▄▄▄▄▄▄▄▄ ▐░█▄▄▄▄▄▄▄▄▄ 
▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░▌       ▐░▌▐░▌▐░░░░░░░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌
▐░█▀▀▀▀▀▀▀▀▀ ▐░█▀▀▀▀█░█▀▀ ▐░▌       ▐░▌▐░▌ ▀▀▀▀▀▀█░▌▐░█▀▀▀▀█░█▀▀ ▐░█▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀█░▌ ▀▀▀▀▀▀▀▀▀█░▌
▐░▌          ▐░▌     ▐░▌  ▐░▌       ▐░▌▐░▌       ▐░▌▐░▌     ▐░▌  ▐░▌                    ▐░▌          ▐░▌
▐░▌          ▐░▌      ▐░▌ ▐░█▄▄▄▄▄▄▄█░▌▐░█▄▄▄▄▄▄▄█░▌▐░▌      ▐░▌ ▐░█▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄█░▌ ▄▄▄▄▄▄▄▄▄█░▌
▐░▌          ▐░▌       ▐░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░▌       ▐░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌
 ▀            ▀         ▀  ▀▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀▀  ▀         ▀  ▀▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀▀                                                                                                       
"""

# 研究進捗
# 2024-04-03, 「昨年の反省」, METHODを加筆。羊の制御モデルの実装を完了。To do 3までを実装。3の細部については要検討。

# 更新履歴：
# 2024-04-02, 初期版を作成開始。
# -03, 研究詳報の加筆。コーディング。


"""
 ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄   ▄▄▄▄▄▄▄▄▄▄▄ 
▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░▌ ▐░░░░░░░░░░░▌
▐░█▀▀▀▀▀▀▀▀▀ ▐░█▀▀▀▀▀▀▀█░▌▐░█▀▀▀▀▀▀▀█░▌▐░█▀▀▀▀▀▀▀▀▀ 
▐░▌          ▐░▌       ▐░▌▐░▌       ▐░▌▐░▌          
▐░▌          ▐░▌       ▐░▌▐░▌       ▐░▌▐░█▄▄▄▄▄▄▄▄▄ 
▐░▌          ▐░▌       ▐░▌▐░▌       ▐░▌▐░░░░░░░░░░░▌
▐░▌          ▐░▌       ▐░▌▐░▌       ▐░▌▐░█▀▀▀▀▀▀▀▀▀ 
▐░▌          ▐░▌       ▐░▌▐░▌       ▐░▌▐░▌          
▐░█▄▄▄▄▄▄▄▄▄ ▐░█▄▄▄▄▄▄▄█░▌▐░█▄▄▄▄▄▄▄█░▌▐░█▄▄▄▄▄▄▄▄▄ 
▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░▌ ▐░░░░░░░░░░░▌
 ▀▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀   ▀▀▀▀▀▀▀▀▀▀▀                                        
"""

"""
██     ███ ███████ ████████ ██   ██  ██████  ██████  
████  ████ ██         ██    ██   ██ ██    ██ ██   ██ 
██ ████ ██ █████      ██    ███████ ██    ██ ██   ██ 
██  ██  ██ ██         ██    ██   ██ ██    ██ ██   ██ 
██      ██ ███████    ██    ██   ██  ██████  ██████  
Font : ANSI Regular                                                      
"""

# 問題設定：
# 境界条件は閉領域。

#%%
# 牧羊犬の制御モデル, 参考文献：[10]
"""
███████ ██   ██ ███████ ███████ ██████  
██      ██   ██ ██      ██      ██   ██ 
███████ ███████ █████   █████   ██████  
     ██ ██   ██ ██      ██      ██      
███████ ██   ██ ███████ ███████ ██                                         
"""

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
# 単体で運用する用のDQNモデル、
"""
███████ ██ ███    ██  ██████  ██      ███████     ██████   ██████  ███    ██ 
██      ██ ████   ██ ██       ██      ██          ██   ██ ██    ██ ████   ██ 
███████ ██ ██ ██  ██ ██   ███ ██      █████       ██   ██ ██    ██ ██ ██  ██ 
     ██ ██ ██  ██ ██ ██    ██ ██      ██          ██   ██ ██ ▄▄ ██ ██  ██ ██ 
███████ ██ ██   ████  ██████  ███████ ███████     ██████   ██████  ██   ████ 
                                                              ▀▀                                                                                                                                                                                
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
        
        #print('state={}'.format(state))
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
# 学習環境の実装。観測、行動、報酬についての設定。
"""
███████ ███    ██ ██    ██ ██ ██████   ██████  ███    ███ ███████ ███    ██ ████████ 
██      ████   ██ ██    ██ ██ ██   ██ ██    ██ ████  ████ ██      ████   ██    ██    
█████   ██ ██  ██ ██    ██ ██ ██████  ██    ██ ██ ████ ██ █████   ██ ██  ██    ██    
██      ██  ██ ██  ██  ██  ██ ██   ██ ██    ██ ██  ██  ██ ██      ██  ██ ██    ██    
███████ ██   ████   ████   ██ ██   ██  ██████  ██      ██ ███████ ██   ████    ██  
"""

import numpy as np
import random
from scipy.spatial import ConvexHull

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
        
        state = np.array([x_star - x_d, y_star - y_d]).reshape(2,)
        #print('state={}, {}'.format(state, state.reshape(2,)))
        return state
    
    def step(self, x_d, y_d, x_s, y_s, action_index):
        x_star, y_star = self.calculate_C(x_d, y_d, x_s, y_s)
        #actions = np.array([[5,0], [0,5], [-5,0], [0,-5], [0,0]])
        actions = np.array([[x_star - x_d, y_star - y_d], [-(x_star - x_d), -(y_star - y_d)], [0, 0]]) # 対象個体に、1.近づく、2.遠ざかる、3.その場にとどまる
        action = actions[action_index]
        #print('index={}, action={}, type={}'.format(action_index, action, type(action)))
        
        x_d_next = x_d + action[0]
        y_d_next = y_d + action[1]
        return x_d_next, y_d_next
    

#%%
# 学習を実行するための各種モジュール。
"""
██      ███████  █████  ██████  ███    ██ ██ ███    ██  ██████  
██      ██      ██   ██ ██   ██ ████   ██ ██ ████   ██ ██       
██      █████   ███████ ██████  ██ ██  ██ ██ ██ ██  ██ ██   ███ 
██      ██      ██   ██ ██   ██ ██  ██ ██ ██ ██  ██ ██ ██    ██ 
███████ ███████ ██   ██ ██   ██ ██   ████ ██ ██   ████  ██████ 
"""

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
        
        self.action_size = 3
        self.state_size = 2
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
        save_dir = 'my_rl_Sheepdog_DQN/'

        # ディレクトリが存在しない場合は作成する
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 1. モデルの保存ディレクトリを指定する
        save_dir = 'my_rl_Sheepdog_DQN/'

        # 2. モデルの重みを取得する
        model = self.agent.qnet.state_dict()
            
        # 4. ファイル名の指定
        file_name = 'Sheepdog_DQN' + tag + '.pth'

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
                
                x_d_next, y_d_next = self.dog.step(x_d, y_d, x_s, y_s, action)
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

folder_name = "sheepdog_DQN_rl_folder"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

file_path = os.path.join(folder_name, "data_reward_sheepdog_DQN_rl1.csv")
data_rewards = np.array(rewards)
np.savetxt(file_path, data_rewards)

file_path = os.path.join(folder_name, "data_sizes_sheepdog_DQN_rl1.csv")
data_file_names = np.array(sizes)
np.savetxt(file_path, data_file_names, fmt='%s')