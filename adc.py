import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
k = 4           # Bits per symbol (M = 2^k = 16)
M = 2**k
n_dim = 2       # 2D constellation (I/Q channels)
batch_size = 512
epochs = 100
learning_rate = 0.005
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transmitter = nn.Sequential(
    nn.Linear(k, 32),
    nn.ReLU(),
    nn.BatchNorm1d(32),
    nn.Linear(32, 32),
    nn.ReLU(),
    nn.Linear(32, n_dim)
).to(device)
receiver = nn.Sequential(
    nn.Linear(n_dim, 32),
    nn.ReLU(),
    nn.BatchNorm1d(32),
    nn.Linear(32, 32),
    nn.ReLU(),
    nn.Linear(32, k)
).to(device)
def normalize_power(x, n_dim):
    avg_power = torch.mean(torch.sum(x**2, dim=1))
    return x * torch.sqrt(torch.tensor(n_dim, dtype=torch.float32) / avg_power)
def pass_awgn_channel(x, snr_db):
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_variance = 1.0 / (2.0 * snr_linear)
    noise = torch.sqrt(torch.tensor(noise_variance, dtype=torch.float32)).to(x.device) * torch.randn_like(x)
    return x + noise
def train_epoch(transmitter, receiver, optimizer, criterion, batch_size, k, n_dim, device):
    transmitter.train()
    receiver.train()
    bits = torch.randint(0, 2, (batch_size, k), dtype=torch.float32).to(device)
    train_snr_db = torch.empty(batch_size, 1).uniform_(3.0, 10.0).to(device)
    optimizer.zero_grad()
    encoded = transmitter(bits)
    transmitted = normalize_power(encoded, n_dim)
    received = pass_awgn_channel(transmitted, train_snr_db)
    outputs = receiver(received)
    loss = criterion(outputs, bits)
    loss.backward()
    optimizer.step()
    return loss.item()
def evaluate_ber_ser(transmitter, receiver, snr_range, k, n_dim, device, total_samples=50000, batch_size=512):
    transmitter.eval()
    receiver.eval()
    print("\nEvaluating BER and SER vs SNR...")
    with torch.no_grad():
        for snr in snr_range:
            bit_errors = 0
            symbol_errors = 0
            batches = total_samples // batch_size
            for _ in range(batches):
                bits = torch.randint(0, 2, (batch_size, k), dtype=torch.float32).to(device)
                encoded = transmitter(bits)
                transmitted = normalize_power(encoded, n_dim)
                received = pass_awgn_channel(transmitted, snr)
                outputs = receiver(received)
                predicted_bits = (outputs > 0.0).float()
                bit_errors += torch.sum(predicted_bits != bits).item()
                # Count symbol errors (if any bit in the symbol is wrong)
                symbol_wrong = torch.sum(predicted_bits != bits, dim=1) > 0
                symbol_errors += torch.sum(symbol_wrong).item()
            ber = bit_errors / (batches * batch_size * k)
            ser = symbol_errors / (batches * batch_size)
            print(f"SNR: {snr:2d} dB | BER: {ber:.6f} | SER: {ser:.5f}")
if __name__ == "__main__":
    optimizer = optim.Adam(
        list(transmitter.parameters()) + list(receiver.parameters()),
        lr=learning_rate
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.BCEWithLogitsLoss()
    print("Training End-to-End Channel Autoencoder (Procedural)...")
    for epoch in range(epochs):
        loss = train_epoch(transmitter, receiver, optimizer, criterion, batch_size, k, n_dim, device)
        scheduler.step(loss)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss:.4f}")
    snr_vals = np.arange(0, 13, 2)
    evaluate_ber_ser(transmitter, receiver, snr_vals, k, n_dim, device)