import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
M = 16
n_dim = 2
batch_size = 512
epochs = 200
steps_per_epoch = 100
learning_rate = 0.01
train_snr_db = 7.0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
w_enc1 = nn.Parameter(torch.randn(32, M, device=device) * np.sqrt(2.0 / M))
b_enc1 = nn.Parameter(torch.zeros(32, device=device))
w_enc2 = nn.Parameter(torch.randn(n_dim, 32, device=device) * np.sqrt(2.0 / 32))
b_enc2 = nn.Parameter(torch.zeros(n_dim, device=device))
w_dec1 = nn.Parameter(torch.randn(32, n_dim, device=device) * np.sqrt(2.0 / n_dim))
b_dec1 = nn.Parameter(torch.zeros(32, device=device))
w_dec2 = nn.Parameter(torch.randn(M, 32, device=device) * np.sqrt(2.0 / 32))
b_dec2 = nn.Parameter(torch.zeros(M, device=device))
parameters = [w_enc1, b_enc1, w_enc2, b_enc2, w_dec1, b_dec1, w_dec2, b_dec2]
optimizer = optim.Adam(parameters, lr=learning_rate)
criterion = nn.CrossEntropyLoss()
def encoder(x):
    h = F.relu(F.linear(x, w_enc1, b_enc1))
    return F.linear(h, w_enc2, b_enc2)
def decoder(x):
    h = F.relu(F.linear(x, w_dec1, b_dec1))
    return F.linear(h, w_dec2, b_dec2)
def normalize_power(x):
    return torch.sqrt(torch.tensor(n_dim, dtype=torch.float32, device=x.device)) * \
           (x / torch.sqrt(torch.mean(torch.sum(x**2, dim=1, keepdim=True))))
def awgn_channel(x, snr_db):
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_variance = 1.0 / (2.0 * snr_linear)
    noise = torch.sqrt(torch.tensor(noise_variance, dtype=torch.float32, device=x.device)) * torch.randn_like(x)
    return x + noise
def forward(x, snr_db):
    encoded = encoder(x)
    normalized = normalize_power(encoded)
    received = awgn_channel(normalized, snr_db)
    decoded = decoder(received)
    return decoded, normalized
print("Training Autoencoder...")
for epoch in range(epochs):
    epoch_loss = 0.0
    for _ in range(steps_per_epoch):
        messages = torch.randint(0, M, (batch_size,)).to(device)
        one_hot = torch.eye(M, device=device)[messages]
        optimizer.zero_grad()
        outputs, _ = forward(one_hot, train_snr_db)
        loss = criterion(outputs, messages)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    if (epoch + 1) % 25 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/steps_per_epoch:.4f}")
with torch.no_grad():
    all_messages = torch.eye(M, device=device)
    _, constellation = forward(all_messages, train_snr_db)
    constellation_np = constellation.cpu().numpy()
plt.figure(figsize=(6, 6))
plt.scatter(constellation_np[:, 0], constellation_np[:, 1], color='red', s=100)
for i in range(M):
    plt.annotate(str(i), (constellation_np[i, 0] + 0.05, constellation_np[i, 1] + 0.05))
plt.axhline(0, color='grey', linestyle='--')
plt.axvline(0, color='grey', linestyle='--')
plt.title(f"Learned 16-ary Constellation (Train SNR = {train_snr_db} dB)")
plt.xlabel("In-Phase (I)")
plt.ylabel("Quadrature (Q)")
plt.grid(True)
plt.show()
snr_range = np.arange(0, 13, 2)
ser_results = []
print("\nEvaluating Symbol Error Rate (SER) vs SNR...")
with torch.no_grad():
    for snr in snr_range:
        errors = 0
        total_samples = 10000
        for _ in range(total_samples // batch_size):
            messages = torch.randint(0, M, (batch_size,)).to(device)
            one_hot = torch.eye(M, device=device)[messages]
            outputs, _ = forward(one_hot, snr)
            predictions = torch.argmax(outputs, dim=1)
            errors += torch.sum(predictions != messages).item()
        ser = errors / total_samples
        ser_results.append(ser)
        print(f"SNR: {snr:2d} dB | SER: {ser:.5f}")