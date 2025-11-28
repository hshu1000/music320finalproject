import numpy as np
import matplotlib.pyplot as plt

fig = None
axes_wave = []
axes_fft = []
lines_wave = []  # list of Line2D for waveforms
lines_fft = []   # list of Line2D for FFTs
num_people = 0


def init_plot():
    global fig
    plt.ion()
    fig = plt.figure(figsize=(10, 6))
    fig.canvas.manager.set_window_title("Realtime Pose Synth Waveforms")
    fig.tight_layout()
    fig.show()


def rebuild_layout(n):
    global fig, axes_wave, axes_fft, lines_wave, lines_fft, num_people

    num_people = n
    fig.clf()

    axes_wave = []
    axes_fft = []
    lines_wave = []
    lines_fft = []

    for i in range(n):
        ax_w = fig.add_subplot(n, 2, 2*i + 1)
        ax_f = fig.add_subplot(n, 2, 2*i + 2)

        # Single line for waveform
        lw_orig, = ax_w.plot([], [], color='tab:blue', label='waveform')

        # Single line for FFT
        lf_orig, = ax_f.plot([], [], color='tab:blue')

        ax_w.set_title(f"Person {i+1} – Waveform")
        ax_f.set_title(f"Person {i+1} – FFT")
        ax_w.legend(loc='upper right')

        axes_wave.append(ax_w)
        axes_fft.append(ax_f)
        lines_wave.append(lw_orig)
        lines_fft.append(lf_orig)

    fig.tight_layout()
    fig.canvas.draw_idle()


def update_plot(wave_list):
    global fig, num_people
    if fig is None:
        return

    n = len(wave_list)
    if n != num_people:
        rebuild_layout(n)

    MAX_FFT = 4000

    for i, wave in enumerate(wave_list):
        wave = np.asarray(wave, dtype=float)
        if len(wave) < 8:
            continue

        wave = wave - np.mean(wave)
        # Convert sample index to time (milliseconds) assuming 44100 Hz sample rate
        duration_ms = len(wave) / 44.1
        x = np.linspace(0, duration_ms, len(wave))

        lw_orig = lines_wave[i]
        lw_orig.set_xdata(x)
        lw_orig.set_ydata(wave)

        axes_wave[i].relim()
        axes_wave[i].autoscale_view()
        axes_wave[i].set_xlabel('Time (ms)')
        axes_wave[i].set_ylim(-0.45, 0.45)

        # FFT (magnitude in dB, relative to waveform peak)
        fft_orig = np.abs(np.fft.rfft(wave))
        freqs = np.fft.rfftfreq(len(wave), 1/44100)
        mask = freqs <= MAX_FFT

        eps = 1e-12
        ref = np.max(fft_orig) + eps
        fft_orig_db = 20.0 * np.log10(fft_orig / ref + eps)

        lf_orig = lines_fft[i]
        lf_orig.set_xdata(freqs[mask])
        lf_orig.set_ydata(fft_orig_db[mask])

        axes_fft[i].set_ylabel('Magnitude (dB)')
        axes_fft[i].set_ylim(-80, 5)
        axes_fft[i].relim()
        axes_fft[i].autoscale_view()

    fig.canvas.draw_idle()
    plt.pause(0.001)
