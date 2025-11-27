import numpy as np
import sounddevice as sd
from scipy.signal import resample
from scipy.interpolate import interp1d
import threading
from synthesizer import Synthesizer

FS = 44100
# Define frequency range
MIN_FREQ = 130.82   # C3
MAX_FREQ = min(2093, FS // 2 - 100.0)  # C7

lock = threading.Lock()
stream = None

# Multiple voices: one period + phase per person
periods = []   # list of np.ndarray (float32)
phases = []    # list of ints

# Reusable synthesizer instance to avoid recreating repeatedly
synth = Synthesizer(sample_rate=FS)


def pose_to_waveform(keypoints):
    # Extract metadata from the end of the list (if present)
    # Metadata is stored as a tuple and should not be used for waveform generation
    if len(keypoints) > 0 and isinstance(keypoints[-1], tuple):
        # Last element is metadata tuple, extract it
        metadata = keypoints[-1]
        pts = np.array(keypoints[:-1], dtype=float)
    else:
        # No metadata, use all points
        metadata = None
        pts = np.array(keypoints, dtype=float)

    center = pts[2]
    rel = pts - center
    rel = rel[np.argsort(rel[:, 0])]

    xs = rel[:, 0]
    ys = -rel[:, 1]
    ys = ys / (np.max(np.abs(ys)) + 1e-9)

    # Add tiny random jitter to x-coordinates to avoid duplicates (which break cubic interpolation)
    # This is imperceptible but allows interpolation to work
    jitter = np.random.normal(0, 1e-10, len(xs))
    xs = xs + jitter

    # Upsample the points 2x for richer waveform control
    # Use cubic interpolation for higher-quality upsampling
    if len(xs) > 1:
        f = interp1d(xs, ys, kind='cubic', fill_value='extrapolate')
        xs_up = np.linspace(xs[0], xs[-1], len(xs) * 2 - 1)
        ys_up = f(xs_up)
        xs = xs_up
        ys = ys_up

    dx = np.abs(np.diff(xs))
    t = np.concatenate([[0], np.cumsum(dx)])
    if t[-1] < 1e-6:
        t = np.linspace(0, 1, len(xs))

    L = max(int(t[-1] * 20), 10)
    tu = np.linspace(0, t[-1], L)
    wave = np.interp(tu, t, ys).astype(np.float32)

    # Compute original spectrum (for reference/plotting)
    # X_orig = np.fft.rfft(wave)

    RAW_MIN = 10
    RAW_MAX = 1000

    raw_width = float(t[-1])
    norm = (RAW_MAX - raw_width) / (RAW_MAX - RAW_MIN)
    norm = np.clip(norm, 0.0, 1.0)

    gamma = 0.4  # tweak this for more/less sensitivity
    v = norm ** gamma

    freq = float(MIN_FREQ * ((MAX_FREQ / MIN_FREQ) ** v))
    freq = float(np.clip(freq, MIN_FREQ, MAX_FREQ))
    print(f"Computed freq: {freq:.2f} Hz from raw width: {raw_width:.4f}")

    # Determine cutoff frequency for lowpass based on avg_y
    # Make the effect more extreme: allow a very low MIN_CUTOFF and a high MAX_CUTOFF
    # Clamp MAX_CUTOFF to a safe value below Nyquist
    MIN_CUTOFF = 300.0
    MAX_CUTOFF = min(FS // 2 - 100.0, 4000.0)
    # extract avg_y and frame height (if provided) from metadata tuple
    avg_y = None
    frame_h = 480  # fallback
    if metadata is not None and isinstance(metadata, (list, tuple)):
        if len(metadata) > 0:
            try:
                avg_y = float(metadata[0])
            except Exception:
                avg_y = None
        if len(metadata) > 1:
            try:
                frame_h = int(metadata[1])
            except Exception:
                frame_h = frame_h

    if avg_y is None:
        cutoff = (MIN_CUTOFF + MAX_CUTOFF) / 2.0
    else:
        # normalize by actual frame height then map to cutoff range
        frame_h = max(1, frame_h)
        norm = np.clip(avg_y / float(frame_h), 0.0, 1.0)
        # Use a stronger exponent to bias values toward the extremes
        POWER = 4.0
        norm_pow = norm ** POWER
        # Use geometric interpolation for perceptual scaling (log-space)
        # cutoff = MIN_CUTOFF * (MAX_CUTOFF / MIN_CUTOFF) ** norm_pow
        cutoff = float(MIN_CUTOFF * ((MAX_CUTOFF / MIN_CUTOFF) ** norm_pow))

    # Apply a steeper lowpass filter to make the effect unmistakable.
    # Increase filter order and apply the filter twice for stronger attenuation.
    try:
        filtered_wave = synth.lowpass_filter(wave, cutoff_hz=cutoff, order=8)
    except Exception:
        # Fall back to a single, safer filter if something goes wrong
        filtered_wave = synth.lowpass_filter(wave, cutoff_hz=cutoff)

    # Also compute spectrum of filtered wave for potential plotting
    # X_filt = np.fft.rfft(filtered_wave)

    # Return original wave, computed freq, filtered wave, and control metadata
    # (cutoff in Hz and normalized control value) so callers can display/debug.
    return wave, freq, filtered_wave, cutoff, norm


def _wave_to_period(wave, freq):
    # Ensure freq stays in playable bounds
    freq = float(np.clip(freq, MIN_FREQ, MAX_FREQ))
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
    # pose_to_waveform now returns (wave, freq, filtered, cutoff, norm)
    _, freq, filtered, _, _ = pose_to_waveform(keypoints)
    update_audio_from_multiple([(filtered, freq)])


def audio_callback(outdata, frames, time, status):
    global periods, phases

    # Check for audio device errors/warnings
    if status:
        print(f"[audio] Status: {status}")

    try:
        # Minimize lock time: copy data once and release immediately
        with lock:
            local_periods = list(periods)  # shallow copy of list
            local_phases = phases.copy()

        # Ensure phase array matches number of periods: pad with zeros or truncate
        if len(local_phases) < len(local_periods):
            local_phases.extend([0] * (len(local_periods) - len(local_phases)))
        elif len(local_phases) > len(local_periods):
            local_phases = local_phases[:len(local_periods)]

        nvoices = len(local_periods)
        if nvoices == 0:
            outdata[:] = 0.0
            return

        out = np.zeros(frames, dtype=np.float32)

        for v, p in enumerate(local_periods):
            try:
                p = np.asarray(p)
                L = len(p)
                if L == 0:
                    continue
                phase = int(local_phases[v]) if v < len(local_phases) else 0
                idxs = (np.arange(frames) + phase) % L
                out += p[idxs]
                local_phases[v] = int((phase + frames) % L)
            except Exception as ex:
                # Skip this voice on error but continue rendering others
                print(f"[audio_callback] voice {v} skipped: {ex}")
                continue

        # Divide by sqrt(nvoices) (coherent-sounding rough normalization) so more voices
        # don't make the mix disproportionately louder.
        if nvoices > 1:
            out /= np.sqrt(nvoices)

        # Peak-limit the mix to a desired maximum without amplifying quiet signals.
        desired_peak = 0.3
        peak = np.max(np.abs(out)) + 1e-9
        if peak > desired_peak:
            out *= (desired_peak / peak)

        # Write the mono mix into the output buffer in a channel-agnostic way
        if outdata.ndim == 1:
            outdata[:] = out
        else:
            nch = outdata.shape[1]
            for ch in range(nch):
                outdata[:, ch] = out

        # Update phase state so the next callback resumes correctly
        with lock:
            phases = local_phases

    except Exception as e:
        print(f"[audio_callback] Error: {e}")
        outdata[:] = 0.0


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