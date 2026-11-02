# End-to-End Learned Communication System

Joint optimization of constellation and receiver over an AWGN channel, using PyTorch.

This project studies whether neural networks can learn digital communication receivers and signal constellations from data. It has two parts:

1. **Learned QPSK receiver.** A fixed Gray-mapped QPSK constellation with a neural decoder. The decoder is compared against the closed-form QPSK bit error rate (BER) at 5 dB and across an Eb/N0 sweep from 0 to 10 dB.
2. **Learned 16-ary autoencoder.** The encoder (constellation), power normalization, AWGN channel, and decoder are trained jointly end to end at 7 dB. The learned two-dimensional constellation and its symbol error rate (SER) are evaluated.

## Key Results

- The learned QPSK receiver matches the theoretical BER curve from 0 to 8 dB to within about 3% at each point. At 5 dB, the learned BER is $5.85\times10^{-3}$ against a theoretical value of $5.95\times10^{-3}$.
- The 16-ary autoencoder converges to a non-uniform constellation with an outer ring of nine points, an inner ring of six points, and one point near the origin, rather than a standard 16-PSK or 16-QAM layout.

## Files

```
.
├── README.md
├── qpsk.py          # QPSK receiver test at 5 dB (two evaluation setups)
├── qpsk_ber.py                   # Learned QPSK BER vs Eb/N0 sweep (0-10 dB)
├── autoencoder_16ary.py       # 16-ary end-to-end autoencoder and SER evaluation
├── project_report.pdf         # Compiled report
├── fig_qpsk_scatter.png       # Figure: QPSK received samples at 5 dB
├── fig_qpsk_ber.png           # Figure: BER vs Eb/N0
└── fig_16ary_constellation.png  # Figure: learned 16-ary constellation
```

## Requirements

- Python 3.9 or newer
- PyTorch
- NumPy
- Matplotlib

Install the dependencies with:

```bash
pip install torch numpy matplotlib
```

A GPU is optional. The autoencoder script uses CUDA automatically if it is available.

## Usage

Run each script from the project folder:

```bash
# 1. QPSK receiver test at 5 dB (about a minute on CPU)
python qpsk.py

# 2. QPSK BER sweep over Eb/N0 from 0 to 10 dB (several minutes on CPU)
python qpsk_ber.py

# 3. 16-ary autoencoder: training, constellation plot, and SER sweep
python autoencoder_16ary.py
```

Each script prints its results to the console and displays its plots.

## Method Summary

**QPSK receiver.** Bits are mapped to Gray QPSK symbols with unit average energy. AWGN is added with $\sigma = 0.5/\sqrt{\gamma_b}$. A two-layer MLP (2 → 64 → 2) is trained with binary cross-entropy for 5000 steps using Adam at learning rate $10^{-3}$. The BER is compared with the closed-form expression $P_b = \tfrac{1}{2}\operatorname{erfc}(\sqrt{E_b/N_0})$.

**16-ary autoencoder.** The encoder (16 → 32 → 2) produces a two-dimensional symbol. Power normalization sets the mean squared symbol norm to 2. The channel adds AWGN at 7 dB during training. The decoder (2 → 32 → 16) outputs logits over the 16 messages. Training uses cross-entropy with Adam at learning rate $10^{-2}$ for 20,000 gradient steps. After training, the constellation is read from the encoder output, and the SER is evaluated over 0 to 12 dB.

## Reproducibility

Each script sets `torch.manual_seed(0)`. Results should reproduce exactly on the same hardware and library versions. Results on GPU may differ slightly from CPU because of floating-point ordering. The reported numbers come from single runs, so no confidence intervals across seeds are included.

## Limitations

- Only the AWGN channel is modeled, with perfect synchronization.
- The autoencoder is trained at a single SNR (7 dB).
- Each experiment uses one random seed.
- The high-$E_b/N_0$ BER points in the sweep rest on few observed errors, so they are imprecise.

## Possible Extensions

- Run the sweep with $10^7$ or more test bits at high $E_b/N_0$.
- Repeat each experiment over several seeds and report confidence intervals.
- Compare the learned 16-point constellation against 16-QAM and 16-PSK at matched average power.
- Train the autoencoder across a range of SNRs.
- Add channel impairments such as phase offset or fading.

## Author

Akshay M Rahul, Dhanush Krishna R N,Pranav S

## License

MIT License
