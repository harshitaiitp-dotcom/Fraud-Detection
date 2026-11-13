import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
class FraudGenerator(nn.Module):
    def __init__(self, noise_dim=32, num_classes=2, output_dim=94):
        super().__init__()
        self.label_embedding = nn.Embedding(num_classes, 16)        
        self.model = nn.Sequential(
            nn.Linear(noise_dim + 16, 128),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Linear(256, output_dim)
        )
    def forward(self, noise, labels):
        emb = self.label_embedding(labels)
        x = torch.cat([noise, emb], dim=1)
        return self.model(x)
class FraudDiscriminator(nn.Module):
    def __init__(self, input_dim=94, num_classes=2):
        super().__init__()
        self.label_embedding = nn.Embedding(num_classes, 16)        
        self.model = nn.Sequential(
            nn.Linear(input_dim + 16, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()  
        )
    def forward(self, features, labels):
        emb = self.label_embedding(labels)
        x = torch.cat([features, emb], dim=1)
        return self.model(x)
def train_generative_model(data_dir="data", epochs=30, batch_size=32):
    classes_path = os.path.join(data_dir, "elliptic_txs_classes.csv")
    features_path = os.path.join(data_dir, "elliptic_txs_features.csv")
    print("Loading data for training...")
    classes_df = pd.read_csv(classes_path)
    features_df = pd.read_csv(features_path, header=None)
    features_df.rename(columns={0: 'txId', 1: 'timestep'}, inplace=True)  
    merged = pd.merge(features_df, classes_df, on='txId')
    labeled = merged[merged['class'].isin(['1', '2'])].copy()
    y = (labeled['class'] == '2').astype(int).values
    X = labeled.iloc[:, 2:96].values.astype(np.float32)
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-6
    X_scaled = (X - mean) / std
    dataset = TensorDataset(torch.tensor(X_scaled), torch.tensor(y, dtype=torch.long))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    noise_dim = 32
    output_dim = 94
    generator = FraudGenerator(noise_dim=noise_dim, num_classes=2, output_dim=output_dim)
    discriminator = FraudDiscriminator(input_dim=output_dim, num_classes=2)
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    criterion = nn.BCELoss()
    print(f"Training Generative AI Model on {len(X)} real transactions across {epochs} epochs...")
    print("-" * 60)
    for epoch in range(1, epochs + 1):
        for real_feats, real_labels in dataloader:
            current_batch_size = real_feats.size(0)
            real_targets = torch.ones(current_batch_size, 1)
            fake_targets = torch.zeros(current_batch_size, 1)
            d_optimizer.zero_grad()
            real_pred = discriminator(real_feats, real_labels)
            d_loss_real = criterion(real_pred, real_targets)
            noise = torch.randn(current_batch_size, noise_dim)
            fake_labels = torch.randint(0, 2, (current_batch_size,))
            fake_feats = generator(noise, fake_labels)
            fake_pred = discriminator(fake_feats.detach(), fake_labels)
            d_loss_fake = criterion(fake_pred, fake_targets)
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()
            g_optimizer.zero_grad()
            gen_pred = discriminator(fake_feats, fake_labels)
            g_loss = criterion(gen_pred, real_targets)
            g_loss.backward()
            g_optimizer.step()
        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch [{epoch:2d}/{epochs:2d}] | Discriminator Loss: {d_loss.item():.4f} | Generator Loss: {g_loss.item():.4f}")
    print("-" * 60)
    print("Training Complete! Now generating synthetic fraud transactions...")
    generator.eval()
    with torch.no_grad():
        num_synthetic = 500
        z = torch.randn(num_synthetic, noise_dim)
        requested_labels = torch.ones(num_synthetic, dtype=torch.long) 
        synthetic_scaled = generator(z, requested_labels).numpy()
        synthetic_unscaled = (synthetic_scaled * std) + mean
    synth_df = pd.DataFrame(synthetic_unscaled)
    synth_df['class'] = '2'  # Illicit
    synth_path = os.path.join(data_dir, "synthetic_illicit_txs.csv")
    synth_df.to_csv(synth_path, index=False)    
    print(f" Successfully generated {num_synthetic} synthetic fraud transactions!")
    print(f" Saved to: {synth_path}")
    print("\nNext: Run '04_evaluate_synthetic.py' to test the quality of your synthetic data!")
if __name__ == "__main__":
    train_generative_model()