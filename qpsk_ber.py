import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
torch.manual_seed(0)
ebn0_db_list = np.arange(0, 11, 1)     # Eb/N0 sweep in dB
train_steps = 5000
batch_size = 1024
lr = 1e-3
num_test_bits = 1_000_000
inv_sqrt2 = 1.0 / np.sqrt(2.0)

def encode_qpsk(bits):
    I = (2.0 * bits[..., 0] - 1.0) * inv_sqrt2
    Q = (2.0 * bits[..., 1] - 1.0) * inv_sqrt2
    return torch.stack([I, Q], dim=-1)
def awgn(x, sigma):
    return x + torch.randn_like(x) * sigma
ber_learned = []
ber_theory = []
for ebn0_db in ebn0_db_list:
    snr = 10.0 ** (ebn0_db / 10.0)
    sigma = 0.5 / np.sqrt(snr)
    decoder = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 2))
    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    for step in range(train_steps):
        bits = torch.randint(0, 2, (batch_size, 2)).float()
        loss = loss_fn(decoder(awgn(encode_qpsk(bits), sigma)), bits)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        bits_test = torch.randint(0, 2, (num_test_bits // 2, 2)).float()
        y = awgn(encode_qpsk(bits_test), sigma)
        ber = ((decoder(y) > 0).float() != bits_test).float().mean().item()
    ber_learned.append(max(ber, 1e-7)) 
    ber_theory.append(0.5 * math.erfc(math.sqrt(snr)))   # Q(sqrt(2 Eb/N0))
    print(f"Eb/N0 {ebn0_db:4.1f} dB | learned BER {ber:.2e} | QPSK theory {ber_theory[-1]:.2e}")
plt.figure(figsize=(8, 6))
plt.semilogy(ebn0_db_list, ber_theory, 'k--', label='QPSK theory')
plt.semilogy(ebn0_db_list, ber_learned, 'o-', color='blue', label='Learned QPSK decoder')
plt.grid(True, which='both', linestyle=':', alpha=0.6)
plt.xlabel('Eb/N0 (dB)')
plt.ylabel('Bit Error Rate (BER)')
plt.title('BER vs Eb/N0: Gray QPSK with Learned Decoder')
plt.legend()
plt.tight_layout()
plt.show()