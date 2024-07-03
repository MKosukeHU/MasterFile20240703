
"""
目標：
VAEのPytorchによる実装の練習。

参考資料：
[1] https://qiita.com/fireflow/items/49e4088b853aedff00f6, 標準的
[2] https://blog.deepblue-ts.co.jp/image-generation/pytorch_vae/, あっさりめ
[3] https://tatsukioike.com/gan/0074/, 割と詳しくまとめられている
[4] https://qiita.com/kenmatsu4/items/b029d697e9995d93aa24, VAEについて理論的な詳説がされている
[5] https://qiita.com/pocokhc/items/912667ef2bbd3f82aa87, VAEを用いたWrold modelについて詳説されている。
[6] https://qiita.com/kenchin110100/items/7ceb5b8e8b21c551d69a, VAEについて歴史を含め解説、他手法との比較も掲載

擬似アルゴリズム：
# 1. データの準備
データセット = データをロード()

# 2. ネットワークの定義
エンコーダ = ネットワークを定義(入力: x, 出力: μ, log(sigma^2))
デコーダ = ネットワークを定義(入力: z, 出力: \hat{x})

# 3. 潜在変数のサンプリング
def サンプリング(μ, log(sigma^2)):
    ε = 標準正規分布からサンプリング()
    z = μ + exp(0.5 * log(sigma^2)) * ε
    return z

# 4. 損失関数の定義
def 損失関数(x, \hat{x}, μ, log(sigma^2)):
    # 再構成誤差 (例: 二乗誤差)
    再構成誤差 = mean(square(x - \hat{x}))

    # KLダイバージェンス
    KLダイバージェンス = -0.5 * sum(1 + log(sigma^2) - square(μ) - exp(log(sigma^2)))

    総損失 = 再構成誤差 + KLダイバージェンス
    return 総損失

# 5. モデルの学習
エポック数 = 100
バッチサイズ = 128
最適化アルゴリズム = Adam()

for エポック in range(エポック数):
    for バッチ in データセットをバッチサイズに分割():
        x = バッチデータ

        # エンコーダの前向き伝播
        μ, log(sigma^2) = エンコーダ(x)

        # 潜在変数のサンプリング
        z = サンプリング(μ, log(sigma^2))

        # デコーダの前向き伝播
        \hat{x} = デコーダ(z)

        # 損失の計算
        損失 = 損失関数(x, \hat{x}, μ, log(sigma^2))

        # 勾配の計算と最適化アルゴリズムによるパラメータの更新
        最適化アルゴリズム.バックプロパゲーション(損失)
        最適化アルゴリズム.パラメータ更新()

    # エポックごとの進捗を出力
    print("エポック:", エポック, "損失:", 損失)

# 6. モデルの評価
テストデータ = テストデータセットをロード()
テスト損失 = 0

for バッチ in テストデータをバッチサイズに分割():
    x = バッチデータ
    μ, log(sigma^2) = エンコーダ(x)
    z = サンプリング(μ, log(sigma^2))
    \hat{x} = デコーダ(z)
    損失 = 損失関数(x, \hat{x}, μ, log(sigma^2))
    テスト損失 += 損失

print("テスト損失:", テスト損失 / テストデータのバッチ数)
"""

#%%
# ライブラリのインポート

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torch.utils as utils
from torchvision import datasets, transforms


#%%
# GPUが使える時は使う

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#%%
# データローダーの準備

def setup_data_loaders(batch_size=128, use_cuda=True):
    root = "../data"
    download=True
    # 画像にどのような前処理を施すか設定します。
    # ToTensor()の他にもサイズを変更するResize()や標準化を行うnormalize()などを指定できます。
    # 画像データをPyTorchが扱えるテンソル形式に変換します。例えば、ピクセル値が0から255の範囲にある画像を、0から1の範囲の値に正規化されたテンソルに変換します。
    trans = transforms.ToTensor()
    # まず、torchvision.datasetsからMNISTという手書き数字のデータセットを読み込みます。
    train_set = datasets.MNIST(root=root, train=True, transform=trans, download=download)
    valid_set = datasets.MNIST(root=root, train=False, transform=trans)
    # データセットをbatch_size個のデータごとに小分けにしたものにして、ミニバッチ学習が可能なようにします。
    # shuffle=True　にすると画像の順序がランダムになったりしますが、ここらへんはどっちでもいいと思います。
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_set, batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader


#%%
# VAEモデルの実装

class VAE(nn.Module): # nn.Moduleクラスを引き継ぎ
    def __init__(self, z_dim, x_dim=28*28):
        super(VAE, self).__init__()
        self.x_dim = x_dim # 入力するデータの次元数
        self.z_dim = z_dim # エンコーダから出力する潜在変数の次元数
        # エンコーダー用の関数
        self.fc1 = nn.Linear(x_dim, 20)
        self.fc2_mean = nn.Linear(20, z_dim) # 平均 mu の出力
        self.fc2_var = nn.Linear(20, z_dim) # 分散 sigma^2 の出力
        # デコーダー用の関数
        self.fc3 = nn.Linear(z_dim, 20)
        self.fc4 = nn.Linear(20, x_dim) # データの再構成
    # エンコーダー
    def encoder(self, x):
        x = x.view(-1, self.x_dim)
        x = F.relu(self.fc1(x))
        mean = self.fc2_mean(x) # 平均
        log_var = self.fc2_var(x) # 分散の対数
        return mean, log_var
    # 潜在ベクトルのサンプリング(再パラメータ化)
    def reparametrizaion(self, mean, log_var, device):
        epsilon = torch.randn(mean.shape, device=device)
        return mean + epsilon*torch.exp(0.5 * log_var) # mu + exp(1/2 * log(sigma^2)) * epsilon
    # デコーダー
    def decoder(self, z):
        y = F.relu(self.fc3(z))
        y = torch.sigmoid(self.fc4(y)) # 各要素にシグモイド関数を適用し、値を(0,1)の範囲に
        return y
    def forward(self, x, device):
        x = x.view(-1, self.x_dim) # 入力データを１次元ベクトル配列に整形
        mean, log_var = self.encoder(x) # 画像xを入力して、エンコーダから平均・分散を出力
        KL = 0.5 * torch.sum(1+log_var - mean**2 - torch.exp(log_var)) # KL[q(z|x)||p(z)]を計算
        z = self.reparametrizaion(mean, log_var, device) # 潜在ベクトルをサンプリング(再パラメータ化)
        x_hat = self.decoder(z) # 潜在ベクトルを入力して、再構築画像 y を出力
        reconstruction = torch.sum(x * torch.log(x_hat+1e-8) + (1 - x) * torch.log(1 - x_hat  + 1e-8)) #E[log p(x|z)], 再構成誤差の計算
        lower_bound = -(KL + reconstruction) #変分下界(ELBO)=E[log p(x|z)] - KL[q(z|x)||p(z)], 総損失関数
        return lower_bound , z, x_hat
    

#%%
# 学習の実行

dataloader_train, dataloader_valid = setup_data_loaders(batch_size=1000) # 学習用（train）と検証用（valid）のデータローダーを作成
model = VAE(z_dim = 1).to(device) # モデルをインスタンス化し、GPUにのせる, .to(device)でGPUを動作
optimizer = optim.Adam(model.parameters(), lr=1e-3) # オプティマイザーとしてAdamを設定。model.parameters() はモデルのすべてのパラメータを返し、lr=1e-3 は学習率
model.train() # モデルを訓練モードに
num_epochs = 200
loss_list = []
for i in range(num_epochs):
    losses = []
    for x, t in dataloader_train: # データローダーからデータを取り出す。
        x = x.to(device) # データをGPUにのせる
        loss, z, y = model(x, device) # 損失関数の値 loss 、潜在ベクトル z 、再構築画像 y を出力
        model.zero_grad() # モデルの勾配を初期化
        loss.backward() # モデル内のパラメータの勾配を計算
        optimizer.step() # 最適化を実行
        losses.append(loss.cpu().detach().numpy()) # ミニバッチの損失を記録
    loss_list.append(np.average(losses)) # バッチ全体の損失を登録
    print("EPOCH: {} loss: {}".format(i, np.average(losses)))


#%%
# 可視化

fig = plt.figure(figsize=(20,4))
model.eval()
zs = []
for x, t in dataloader_valid:
    for i, im in enumerate(x.view(-1,28,28).detach().numpy()[:10]):
        # 元画像を可視化
        ax = fig.add_subplot(2, 10, i+1, xticks=[], yticks=[])
        ax.imshow(im, "gray")
    x = x.to(device)
    _, _, y = model(x, device) #再構築画像 y を出力
    y  = y.view(-1,28,28)
    for i, im in enumerate(y.cpu().detach().numpy()[:10]):
        # 再構築画像を可視化
        ax = fig.add_subplot(2,10,11+i, xticks=[], yticks=[])
        ax.imshow(im, "gray")


#%%