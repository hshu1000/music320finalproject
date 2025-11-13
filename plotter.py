import numpy as np
import matplotlib.pyplot as plt

fig = None
axes_wave = []
axes_fft = []
lines_wave = []
lines_fft = []
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

        lw, = ax_w.plot([], [])
        lf, = ax_f.plot([], [])

        ax_w.set_title(f"Person {i+1} – Waveform")
        ax_f.set_title(f"Person {i+1} – FFT")

        axes_wave.append(ax_w)
        axes_fft.append(ax_f)
        lines_wave.append(lw)
        lines_fft.append(lf)

    fig.tight_layout()
    fig.canvas.draw_idle()


def update_plot(wave_list):
    global fig, num_people

    if fig is None:
        return

    n = len(wave_list)

    if n != num_people:
        rebuild_layout(n)

    for i, wave in enumerate(wave_list):
        wave = np.asarray(wave, dtype=float)
        if len(wave) == 0:
            continue

        x = np.arange(len(wave))
        lines_wave[i].set_xdata(x)
        lines_wave[i].set_ydata(wave)
        axes_wave[i].relim()
        axes_wave[i].autoscale_view()

        fft_vals = np.abs(np.fft.rfft(wave))
        freqs = np.fft.rfftfreq(len(wave), 1/44100)
        lines_fft[i].set_xdata(freqs)
        lines_fft[i].set_ydata(fft_vals)
        axes_fft[i].relim()
        axes_fft[i].autoscale_view()

    fig.canvas.draw_idle()
    plt.pause(0.001)
