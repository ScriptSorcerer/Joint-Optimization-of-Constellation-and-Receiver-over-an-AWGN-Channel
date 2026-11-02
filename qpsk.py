import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
torch.manual_seed(0)
num_bits = 100000
ebn0_db = 5.0
ebn0_linear = 10.0 ** (ebn0_db / 10.0)
sigma = 0.5 / np.sqrt(ebn0_linear)
train_steps = 5000
batch_size = 1024
lr = 1e-3
msgs = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
def encode_qpsk(bits):
    I = (2.0 * bits[..., 0] - 1.0) / np.sqrt(2.0)
    Q = (2.0 * bits[..., 1] - 1.0) / np.sqrt(2.0)
    return torch.stack([I, Q], dim=-1)
decoder = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 2))
def awgn(x):
    return x + torch.randn_like(x) * sigma
optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
loss_fn = nn.BCEWithLogitsLoss()
for step in range(train_steps):
    bits = torch.randint(0, 2, (batch_size, 2)).float()
    loss = loss_fn(decoder(awgn(encode_qpsk(bits))), bits)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 1000 == 0 or step == train_steps - 1:
        print(f"step {step:5d}  loss {loss.item():.4f}")
with torch.no_grad():
    bits_test = torch.randint(0, 2, (num_bits // 2, 2)).float()
    y_test = awgn(encode_qpsk(bits_test))
    ber_setup1 = ((decoder(y_test) > 0).float() != bits_test).float().mean().item()
with torch.no_grad():
    codewords = encode_qpsk(msgs)                         # (4, 2) clean QPSK symbols
    msg_idx = torch.randint(0, 4, (num_bits // 2,))       # random 2-bit messages
    bits_direct = msgs[msg_idx]
    y_direct = awgn(codewords[msg_idx])                   # AWGN directly on signal
    bits_direct_hat = (decoder(y_direct) > 0).float()
    ber_setup2 = (bits_direct_hat != bits_direct).float().mean().item()
    decoded_msg = (bits_direct_hat * torch.tensor([2, 1.])).sum(-1).long()
print("\n--- Simulation Results (QPSK) ---")
print(f"Setup 1 BER: {ber_setup1:.6f}")
print(f"Setup 2 BER: {ber_setup2:.6f}")
cw = codewords.numpy()
y1 = y_test.numpy()
y_np = y_direct.numpy()
dec_np = decoded_msg.numpy()
with torch.no_grad():
    dec_idx = ((decoder(y_test) > 0).float() @ torch.tensor([2.0, 1.0])).long().numpy()
msg_labels = [f"{int(m[0])}{int(m[1])}" for m in msgs]
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.scatter(y1[:, 0], y1[:, 1], c=dec_idx, cmap='tab10', alpha=0.4, s=8,
            label='Output from receiver (color = decoded bits)')
plt.scatter(cw[:, 0], cw[:, 1], color='red', s=80, zorder=5,
            label='QPSK Constellation Points (4 points)')
for x, y, label in zip(cw[:, 0], cw[:, 1], msg_labels):
    plt.text(x + 0.04, y + 0.04, label, fontsize=9, fontweight='bold', color='darkred', zorder=6)
plt.axhline(0, color='grey', linestyle='--', linewidth=1)
plt.axvline(0, color='grey', linestyle='--', linewidth=1)
plt.xlim(-2, 2); plt.ylim(-2, 2)
plt.xlabel('In-Phase (I)'); plt.ylabel('Quadrature (Q)')
plt.title(f'1. QPSK Modulator -> AWGN -> Decoder\n(BER: {ber_setup1:.5f})', fontweight='bold')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper right', fontsize=8)
plt.subplot(1, 2, 2)
plt.scatter(y_np[:, 0], y_np[:, 1], c=dec_np, cmap='tab10', alpha=0.4, s=8,
            label='Output from receiver (color = decoded bits)')
plt.scatter(cw[:, 0], cw[:, 1], color='black', s=80, zorder=5,
            label='QPSK Constellation Points (4 points)')
for x, y, label in zip(cw[:, 0], cw[:, 1], msg_labels):
    plt.text(x + 0.04, y + 0.04, label, fontsize=9, fontweight='bold', color='black', zorder=6)
plt.axhline(0, color='grey', linestyle='--', linewidth=1)
plt.axvline(0, color='grey', linestyle='--', linewidth=1)
plt.xlim(-2, 2); plt.ylim(-2, 2)
plt.xlabel('In-Phase (I)'); plt.ylabel('Quadrature (Q)')
plt.title(f'2. AWGN Directly into Trained Decoder\n(BER: {ber_setup2:.5f})', fontweight='bold')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper right', fontsize=8)
plt.tight_layout()
plt.show()