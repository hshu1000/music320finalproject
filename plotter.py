import numpy as np
import matplotlib.pyplot as plt

fig = None
axes_wave = []
axes_fft = []
lines_wave = []  # list of tuples: (orig_line, filtered_line)
lines_fft = []   # list of tuples: (orig_line, filtered_line)
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

        # create two lines per plot: original (blue) and filtered (orange)
        lw_orig, = ax_w.plot([], [], color='tab:blue', label='orig')
        lw_filt, = ax_w.plot([], [], color='tab:orange', linewidth=2.0, linestyle='--', alpha=0.9, label='lowpass')

        lf_orig, = ax_f.plot([], [], color='tab:blue')
        lf_filt, = ax_f.plot([], [], color='tab:orange')

        ax_w.set_title(f"Person {i+1} – Waveform")
        ax_f.set_title(f"Person {i+1} – FFT")
        ax_w.legend(loc='upper right')

        axes_wave.append(ax_w)
        axes_fft.append(ax_f)
        lines_wave.append((lw_orig, lw_filt))
        lines_fft.append((lf_orig, lf_filt))

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

    for i, entry in enumerate(wave_list):
        # entry can be either a single waveform (legacy) or a tuple (orig, filtered)
        if isinstance(entry, (list, tuple)) and len(entry) == 2:
            orig_wave = np.asarray(entry[0], dtype=float)
            filt_wave = np.asarray(entry[1], dtype=float)
        else:
            orig_wave = np.asarray(entry, dtype=float)
            filt_wave = None

        if len(orig_wave) < 8:
            continue

        orig_wave = orig_wave - np.mean(orig_wave)
        # Convert sample index to time (milliseconds) assuming 44100 Hz sample rate
        duration_ms = len(orig_wave) / 44.1
        x = np.linspace(0, duration_ms, len(orig_wave))

        lw_orig, lw_filt = lines_wave[i]
        lw_orig.set_xdata(x)
        lw_orig.set_ydata(orig_wave)

        if filt_wave is not None and len(filt_wave) >= 8:
            filt_wave = filt_wave - np.mean(filt_wave)
            duration_filt_ms = len(filt_wave) / 44.1
            x_filt = np.linspace(0, duration_filt_ms, len(filt_wave))
            lw_filt.set_xdata(x_filt)
            lw_filt.set_ydata(filt_wave)
        else:
            lw_filt.set_xdata([])
            lw_filt.set_ydata([])

        axes_wave[i].relim()
        axes_wave[i].autoscale_view()
        axes_wave[i].set_xlabel('Time (ms)')
        # Fix y-axis for stable viewing
        axes_wave[i].set_ylim(-0.45, 0.45)

        # FFTs (display magnitude in dB, relative to the original waveform peak)
        fft_orig = np.abs(np.fft.rfft(orig_wave))
        freqs = np.fft.rfftfreq(len(orig_wave), 1/44100)
        mask = freqs <= MAX_FFT

        # Reference peak for dB scaling (avoid divide-by-zero)
        eps = 1e-12
        ref = np.max(fft_orig) + eps
        fft_orig_db = 20.0 * np.log10(fft_orig / ref + eps)

        lf_orig, lf_filt = lines_fft[i]
        lf_orig.set_xdata(freqs[mask])
        lf_orig.set_ydata(fft_orig_db[mask])

        if filt_wave is not None and len(filt_wave) >= 8:
            fft_filt = np.abs(np.fft.rfft(filt_wave))
            fft_filt_db = 20.0 * np.log10(fft_filt / ref + eps)
            lf_filt.set_xdata(freqs[mask])
            lf_filt.set_ydata(fft_filt_db[mask])
        else:
            lf_filt.set_xdata([])
            lf_filt.set_ydata([])

        # Set FFT axis labels and limits in dB for clearer comparison
        axes_fft[i].set_ylabel('Magnitude (dB)')
        axes_fft[i].set_ylim(-80, 5)
        axes_fft[i].relim()
        axes_fft[i].autoscale_view()

    fig.canvas.draw_idle()
    plt.pause(0.001)
