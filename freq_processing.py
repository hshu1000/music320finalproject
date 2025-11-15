import numpy as np
import sounddevice as sd
from scipy.signal import resample
import threading

FS = 44100
MIN_FREQ = 261.62

lock = threading.Lock()
stream = None

# multiple voices: one period + phase per person
periods = []   # list of np.ndarray (float32)
phases = []    # list of ints


def pose_to_waveform(keypoints):
    pts = np.array(keypoints, dtype=float)
    center = pts[2]
    rel = pts - center
    rel = rel[np.argsort(rel[:, 0])]

    xs = rel[:, 0]
    ys = -rel[:, 1]
    ys = ys / (np.max(np.abs(ys)) + 1e-9)

    dx = np.abs(np.diff(xs))
    t = np.concatenate([[0], np.cumsum(dx)])
    if t[-1] < 1e-6:
        t = np.linspace(0, 1, len(xs))

    L = max(int(t[-1] * 20), 10)
    tu = np.linspace(0, t[-1], L)
    wave = np.interp(tu, t, ys).astype(np.float32)

    raw_width = float(t[-1])
    alpha = 0.4
    scaled = raw_width**alpha

    BASE = 440.0
    SCALE = 30.0

    freq = BASE / (1.0 + scaled / SCALE)
    freq = float(np.clip(freq, MIN_FREQ, 2000.0))

    return wave, freq


def _wave_to_period(wave, freq):
    freq = float(np.clip(freq, MIN_FREQ, 2000.0))
    ps = max(32, int(FS / freq))
    p = resample(wave, ps).astype(np.float32)
    p /= (np.max(np.abs(p)) + 1e-6)
    p *= 0.3
    return p


def update_audio_from_multiple(wave_freq_list):
    global periods, phases

    new_periods = []
    for wave, freq in wave_freq_list:
        p = _wave_to_period(wave, freq)
        if len(p) > 0:
            new_periods.append(p)

    with lock:
        periods = new_periods
        phases = [0] * len(new_periods)


def update_audio_from_pose(keypoints):
    wave, freq = pose_to_waveform(keypoints)
    update_audio_from_multiple([(wave, freq)])


def audio_callback(outdata, frames, time, status):
    global periods, phases

    with lock:
        local_periods = periods
        local_phases = phases.copy()

    nvoices = len(local_periods)
    if nvoices == 0:
        outdata[:] = 0.0
        return

    out = np.zeros(frames, dtype=np.float32)

    for v, p in enumerate(local_periods):
        L = len(p)
        if L == 0:
            continue
        phase = local_phases[v]
        idxs = (np.arange(frames) + phase) % L
        out += p[idxs]
        local_phases[v] = (phase + frames) % L

    out /= max(1, nvoices)
    max_abs = np.max(np.abs(out)) + 1e-6
    out = 0.3 * out / max_abs

    with lock:
        phases = local_phases

    outdata[:, 0] = out
    if outdata.shape[1] > 1:
        outdata[:, 1] = out


def start_audio_thread():
    global stream
    if stream is not None:
        return
    stream = sd.OutputStream(
        channels=1,
        samplerate=FS,
        callback=audio_callback,
        blocksize=512,
        dtype='float32'
    )
    stream.start()