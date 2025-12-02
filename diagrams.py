import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

# Config
AUDIO_FILE   = "recordings/piano.wav" 
DPI          = 300
N_FFT        = 2048
HOP_LENGTH   = 256

y, sr = librosa.load(AUDIO_FILE, sr=None)

S = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)

plt.figure(figsize=(16, 9))

librosa.display.specshow(
    S_db,
    sr=sr,
    hop_length=HOP_LENGTH,
    x_axis="time",
    y_axis="hz",
    cmap="magma"
)

plt.colorbar(format="%+2.0f dB")
plt.title("Piano Spectrogram")
plt.ylim(0,10000)
plt.show()
