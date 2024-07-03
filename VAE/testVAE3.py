
#%%
# 画像読み込みと前処理
from PIL import Image
import os
import numpy as np
import torch
from torchvision import transforms

def load_images(image_dir):
    """
    画像ディレクトリから画像を読み込んでnumpy配列に変換する関数

    Parameters:
    image_dir (str): 画像が保存されているディレクトリ

    Returns:
    numpy.ndarray: 画像データセット
    """
    images = []
    for filename in os.listdir(image_dir):
        if filename.endswith(".png"):
            img_path = os.path.join(image_dir, filename)
            image = Image.open(img_path).convert('L')  # グレースケールに変換
            image = image.resize((28, 28))  # サイズを28x28にリサイズ
            image = np.array(image) / 255.0  # 正規化
            images.append(image)
    images = np.array(images)
    images = np.expand_dims(images, axis=1)  # チャネル次元を追加
    return images

# 使用例
image_dir = 'vae_training_images'
images = load_images(image_dir)
print(images.shape)  # (num_images, 1, 28, 28)


#%%
# データローダの作成

from torch.utils.data import DataLoader, Dataset

class ImageDataset(Dataset):
    def __init__(self, images):
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]

# データセットの作成
dataset = ImageDataset(images)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)


#%%
# VAEで学習

import torch
import torch.nn as nn
import torch.optim as optim

class VAE(nn.Module):
    def __init__(self, image_size=784, hidden_dim=256, latent_dim=64):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(image_size, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)
        self.fc22 = nn.Linear(hidden_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, image_size)

    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = torch.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 学習
num_epochs = 90
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(data_loader):
        data = data.to(device, dtype=torch.float32)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    print(f'Epoch {epoch + 1}, Loss: {train_loss / len(data_loader.dataset)}')

# 学習したモデルの保存
torch.save(model.state_dict(), 'testVAE3_model.pth')


#%%
# モデルテスト

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class VAE(nn.Module):
    def __init__(self, image_size=784, hidden_dim=256, latent_dim=64):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(image_size, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)
        self.fc22 = nn.Linear(hidden_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, image_size)

    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = torch.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def load_model(model_path, device):
    model = VAE().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def plot_reconstructed_images(original, reconstructed, n=10):
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # Original images
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(original[i].cpu().numpy().reshape(28, 28), cmap='gray')
        plt.title("Original")
        plt.axis("off")
        
        # Reconstructed images
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(reconstructed[i].cpu().detach().numpy().reshape(28, 28), cmap='gray')
        plt.title("Reconstructed")
        plt.axis("off")
    plt.show()

# テストデータセットの準備
test_images = load_images(image_dir)
test_dataset = ImageDataset(test_images)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True)

# 学習済みモデルのロード
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = 'testVAE3_model.pth'
model = load_model(model_path, device)

# テストデータでモデルを評価
model.eval()
with torch.no_grad():
    for batch_idx, data in enumerate(test_loader):
        data = data.to(device, dtype=torch.float32)
        reconstructed, _, _ = model(data)
        plot_reconstructed_images(data, reconstructed)
        break  # 最初のバッチだけを表示

# %%
