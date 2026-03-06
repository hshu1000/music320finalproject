import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


csv_file = "analysis_scripts/hand_mode_sin_approx.csv"  
A = 0.3 
T = 1.75e-3   
f0 = 1 / T  
num_repeats_for_fft_plot = 64
num_harmonics_for_thd = 10        
fft_plot_max_freq = 20000   

df = pd.read_csv(csv_file)
t_ms = df["time_ms"].to_numpy()
x = df["amplitude"].to_numpy()

idx = np.argsort(t_ms)
t_ms = t_ms[idx]
x = x[idx]

t = t_ms * 1e-3
dt = np.mean(np.diff(t))
fs = 1 / dt
N = len(x)

print(f"N = {N}")
print(f"dt = {dt:.6e} s")
print(f"fs = {fs:.3f} Hz")
print(f"f0 = {f0:.3f} Hz")

s_ref = A * np.sin(2 * np.pi * f0 * t)

# full cross-correlation
corr = np.correlate(x - np.mean(x), s_ref - np.mean(s_ref), mode="full")
lags = np.arange(-N + 1, N)
best_lag = lags[np.argmax(corr)]

# shift sine by best_lag samples
s_aligned = np.roll(s_ref, best_lag)

# Calculate error metrics
mse = np.mean((x - s_aligned) ** 2)
rmse = np.sqrt(mse)

print("\n=== Sine wave comparison ===")
print(f"Best lag (samples) = {best_lag}")
print(f"MSE   = {mse:.8e}")
print(f"RMSE  = {rmse:.8e}")

# Compute THD
x_ac = x - np.mean(x)

X = np.fft.rfft(x_ac)
freqs = np.fft.rfftfreq(N, d=dt)
amps_peak = 2 * np.abs(X) / N
A1 = amps_peak[1]

harmonic_numbers = np.arange(1, min(num_harmonics_for_thd + 1, len(amps_peak)))
harmonic_freqs = harmonic_numbers * f0
harmonic_amps = amps_peak[harmonic_numbers]

# THD uses harmonics 2..n divided by harmonic 1
if A1 <= 0:
    raise ValueError("Fundamental amplitude is zero. Cannot compute THD.")

thd = np.sqrt(np.sum(harmonic_amps[1:] ** 2)) / A1
thd_percent = 100 * thd

print("\n=== THD ===")
print(f"Fundamental amplitude V1 = {A1:.8e}")
print(f"THD (harmonics 2 to {harmonic_numbers[-1]}) = {thd:.8e}")
print(f"THD% = {thd_percent:.4f}%")

print("\nFirst 10 harmonics:")
print("n   freq(Hz)      amplitude")
for n, f, a in zip(harmonic_numbers, harmonic_freqs, harmonic_amps):
    print(f"{n:<2d}  {f:10.3f}   {a:.8e}")

# Repeat waveform for display
x_rep = np.tile(x_ac, num_repeats_for_fft_plot)
N_rep = len(x_rep)

X_rep = np.fft.rfft(x_rep)
freqs_rep = np.fft.rfftfreq(N_rep, d=dt)
amps_rep_peak = 2 * np.abs(X_rep) / N_rep

# Plots
plt.figure(figsize=(8, 4))
plt.plot(t_ms, x, label="Hand mode PBW")
plt.plot(t_ms, s_aligned, "--", label="Reference signal")
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude")
plt.title("Hand mode PBW vs. Reference signal")
plt.legend()
plt.tight_layout()

plt.figure(figsize=(8, 4))
plt.stem(harmonic_freqs, harmonic_amps, basefmt=" ")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Peak amplitude")
plt.title("First 10 Harmonics")
plt.tight_layout()

plt.figure(figsize=(9, 4))
mask = freqs_rep <= fft_plot_max_freq
plt.plot(freqs_rep[mask], amps_rep_peak[mask])
plt.xlabel("Frequency (Hz)")
plt.ylabel("Peak amplitude")
plt.title(f"FFT Magnitude")
plt.tight_layout()

plt.show()