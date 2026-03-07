import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

# Config
AUDIO_FILE   = "recordings/reference_violin.m4a" 
DPI          = 300
N_FFT        = 2048
HOP_LENGTH   = 256

fs = 44.1e3
y, sr = librosa.load(AUDIO_FILE, sr=None)

y = y[int(3*fs):int(5*fs)]
S = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)

plt.figure(figsize=(10, 8))

librosa.display.specshow(
    S_db,
    sr=sr,
    hop_length=HOP_LENGTH,
    x_axis="time",
    y_axis="hz",
    cmap="magma"
)

plt.colorbar(format="%+2.0f dB")
plt.title("Violin Spectrogram")
plt.ylim(0,10000)
plt.show()
