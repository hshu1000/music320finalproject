import numpy as np
import sounddevice as sd
from scipy.signal import resample
import threading

FS = 44100
MIN_FREQ = 261.62

current_period = np.zeros(512, dtype=np.float32)
current_idx = 0
lock = threading.Lock()
stream = None


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


def audio_callback(outdata, frames, time, status):
    global current_period, current_idx

    with lock:
        p = current_period
        idx = current_idx

    n = len(p)
    if n == 0:
        outdata[:] = 0
        return

    out = np.empty(frames, dtype=np.float32)
    for i in range(frames):
        out[i] = p[idx]
        idx += 1
        if idx >= n:
            idx = 0

    with lock:
        current_idx = idx

    outdata[:, 0] = out


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


def update_audio_from_pose(keypoints):
    global current_period, current_idx
    wave, freq = pose_to_waveform(keypoints)
    p = _wave_to_period(wave, freq)

    with lock:
        current_period = p
        if current_idx >= len(p):
            current_idx = 0
