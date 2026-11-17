import argparse 
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.special import sph_harm

# =========================
# 1. 3D 電子雲データ生成
# =========================

def electron_cloud_3d(n, l, m, size=32):
    x = np.linspace(-10, 10, size)
    y = np.linspace(-10, 10, size)
    z = np.linspace(-10, 10, size)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    R = np.sqrt(X**2 + Y**2 + Z**2) + 1e-6
    Theta = np.arccos(Z / R)
    Phi = np.arctan2(Y, X)

    Y_lm = np.real(sph_harm(m, l, Phi, Theta))

    R_n = np.exp(-R / n) * (R ** l)

    psi = R_n * Y_lm
    prob = psi**2

    prob -= prob.min()
    prob /= prob.max()

    return prob.astype(np.float32)

def generate_dataset(n_samples=800, size=32):
    data = []
    for _ in range(n_samples):
        n = np.random.randint(1, 5)
        l = np.random.randint(0, n)
        m = np.random.randint(-l, l + 1)
        cloud = electron_cloud_3d(n, l, m, size)
        data.append(cloud)
    data = np.array(data)
    data = data[:, None, :, :, :]  # (N,1,D,H,W)
    return data

# =========================
# 2. Volume VAE
# =========================

class VolumeVAE(nn.Module):
    def __init__(self, latent_dim=2, size=32):
        super().__init__()
        self.size = size

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(size**3, 512),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, size**3),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder(z)
        return x.view(-1, 1, self.size, self.size, self.size)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# =========================
# 3. 損失関数（β-VAE）
# =========================

def loss_fn(recon, x, mu, logvar, beta=5.0):
    recon_loss = nn.functional.mse_loss(recon, x)
    kl = -0.5 * torch.mean(1 + logvar - mu**2 - logvar.exp())
    return recon_loss + beta * kl

# =========================
# 4. 学習
# =========================

def train_vae():
    print("3D電子雲データ生成中...")
    data = generate_dataset()
    dataset = torch.tensor(data)

    vae = VolumeVAE(latent_dim=2, size=32)
    optimizer = optim.Adam(vae.parameters(), lr=1e-3)

    epochs = 25
    batch_size = 32

    print("学習開始")
    for epoch in range(epochs):
        perm = torch.randperm(len(dataset))
        total_loss = 0.0

        for i in range(0, len(dataset), batch_size):
            batch = dataset[perm[i:i+batch_size]]
            recon, mu, logvar = vae(batch)
            loss = loss_fn(recon, batch, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}  Loss: {total_loss:.4f}")

    return vae

# =========================
# 5. Slider GUI（Zスライス）
# =========================

def slider_gui(vae, size=32):
    vae.eval()

    z1_init, z2_init = 0.0, 0.0
    slice_init = size // 2

    z = torch.tensor([[z1_init, z2_init]], dtype=torch.float32)
    vol = vae.decode(z).detach().numpy()[0, 0]

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.3)

    img = ax.imshow(vol[:, :, slice_init], cmap="inferno")
    ax.set_title("Electron Cloud Volume VAE")
    ax.axis("off")

    ax_z1 = plt.axes([0.15, 0.18, 0.7, 0.03])
    ax_z2 = plt.axes([0.15, 0.13, 0.7, 0.03])
    ax_slice = plt.axes([0.15, 0.07, 0.7, 0.03])

    s_z1 = Slider(ax_z1, "z1", -4, 4, valinit=z1_init)
    s_z2 = Slider(ax_z2, "z2", -4, 4, valinit=z2_init)
    s_slice = Slider(ax_slice, "slice", 0, size - 1, valinit=slice_init, valstep=1)

    def update(val):
        z = torch.tensor([[s_z1.val, s_z2.val]], dtype=torch.float32)
        vol = vae.decode(z).detach().numpy()[0, 0]
        img.set_data(vol[:, :, int(s_slice.val)])
        fig.canvas.draw_idle()

    s_z1.on_changed(update)
    s_z2.on_changed(update)
    s_slice.on_changed(update)

    plt.show()

# =========================
# 6. 引数
# =========================

def arg_parse():
    parser = argparse.ArgumentParser(description='3D電子雲 × ボリュームVAE × Slider GUI')

    parser.add_argument('--train', action='store_true')

    args = parser.parse_args()

    return args

# =========================
# 7. メイン
# =========================

if __name__ == "__main__":
    args = arg_parse()

    if args.train == True:
        vae = train_vae()
        torch.save(vae.state_dict(), "model.pt")
    else:
        vae = VolumeVAE(latent_dim=2, size=32)
        vae.load_state_dict(torch.load("model.pt", map_location="cpu"))
        slider_gui(vae)
