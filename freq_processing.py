import numpy as np
import sounddevice as sd
from scipy.signal import resample
from scipy.interpolate import interp1d
from scipy import signal
from scipy.io import wavfile
import threading
import math
import os

FS = 44100

# Pedal effect
PEDAL_MODE = False          # True = sustain using echo/reverb-like tail
PEDAL_TIME = 10.0           # approximate sustain time in seconds

REVERB_DELAY_SEC = 0.3      # base delay for echo taps (in seconds)
# Reverb internal state (simple feedback comb)
reverb_feedback = 0.8
reverb_buffer = np.zeros(int(FS * REVERB_DELAY_SEC), dtype=np.float32)
reverb_idx = 0


def _update_reverb_params():
    """
    Choose feedback so that the echo decays to about 1% after PEDAL_TIME seconds.
    """
    global reverb_feedback
    if PEDAL_TIME <= 0.0:
        reverb_feedback = 0.0
        return
    D = REVERB_DELAY_SEC
    T = PEDAL_TIME
    # feedback^(T/D) ≈ 0.01  => feedback = 0.01^(D/T)
    reverb_feedback = 0.01 ** (D / max(T, 1e-3))


_update_reverb_params()


def set_pedal_mode(on: bool):
    """
    Turn sustain/echo pedal mode on or off.
    When turned off, clear any remaining tail.
    """
    global PEDAL_MODE, reverb_buffer, reverb_idx
    PEDAL_MODE = bool(on)
    if not PEDAL_MODE:
        # hard-cut any remaining tail
        reverb_buffer[:] = 0.0
        reverb_idx = 0
    print(f'[pedal] Pedal mode {"ON" if PEDAL_MODE else "OFF"}')


def set_pedal_time(seconds: float):
    """
    Set approximate sustain time for pedal tail (in seconds).
    """
    global PEDAL_TIME
    try:
        seconds = float(seconds)
    except Exception:
        print('[pedal] Invalid time, must be a number.')
        return
    PEDAL_TIME = max(0.5, seconds)
    _update_reverb_params()
    print(f'[pedal] sustain time set to {PEDAL_TIME:.2f} s')


def process_reverb_block(x: np.ndarray) -> np.ndarray:
    """
    Streaming echo / reverb-like effect used for pedal mode.
    - When PEDAL_MODE is False, returns x unchanged.
    - When True, uses a feedback comb to create a tail that
      keeps ringing even when input becomes zero.
    """
    global reverb_buffer, reverb_idx, reverb_feedback

    if not PEDAL_MODE or reverb_feedback <= 0.0:
        return x

    x = np.asarray(x, dtype=np.float32)
    y = np.zeros_like(x)

    buf = reverb_buffer
    idx = reverb_idx
    L = len(buf)

    for n in range(len(x)):
        delayed = buf[idx]
        # mix delayed content with dry signal
        y[n] = x[n] + delayed
        # update buffer with new feedback
        buf[idx] = x[n] + delayed * reverb_feedback
        idx += 1
        if idx == L:
            idx = 0

    reverb_idx = idx

    peak = np.max(np.abs(y)) + 1e-9
    if peak > 1.0:
        y *= 1.0 / peak
    return y


# Flanger
FLANGER_ON = False
FLANGER_RATE = 0.7          # Hz (LFO rate)
FLANGER_DEPTH_MS = 5.0      # ms modulation depth
FLANGER_BASE_DELAY_MS = 2.0 # ms base delay

MAX_FLANGER_DELAY_MS = 10.0
FLANGER_BUFFER_SIZE = int(FS * MAX_FLANGER_DELAY_MS / 1000.0) + 2048
flanger_buffer = np.zeros(FLANGER_BUFFER_SIZE, dtype=np.float32)
flanger_idx = 0
flanger_phase = 0.0


def set_flanger_on(on: bool):
    global FLANGER_ON
    FLANGER_ON = bool(on)
    print(f'[flanger] {"ON" if FLANGER_ON else "OFF"}')


def set_flanger_params(rate=None, depth_ms=None):
    global FLANGER_RATE, FLANGER_DEPTH_MS
    if rate is not None:
        try:
            r = float(rate)
            FLANGER_RATE = max(0.01, min(5.0, r))
        except Exception:
            print('[flanger] Invalid rate value.')
    if depth_ms is not None:
        try:
            d = float(depth_ms)
            FLANGER_DEPTH_MS = max(0.1, min(5.0, d))
        except Exception:
            print('[flanger] Invalid depth value.')

    print(f'[flanger] rate={FLANGER_RATE:.2f} Hz, depth={FLANGER_DEPTH_MS:.2f} ms')


def apply_flanger_block(x: np.ndarray) -> np.ndarray:
    """
    Simple streaming flanger:
    y[n] = x[n] + 0.5 * x[n - delay(n)]
    delay(n) is modulated by a sine LFO.
    """
    global flanger_buffer, flanger_idx, flanger_phase

    if not FLANGER_ON:
        return x

    x = np.asarray(x, dtype=np.float32)
    y = np.zeros_like(x)

    base_delay = int(FLANGER_BASE_DELAY_MS * FS / 1000.0)
    depth_samples = int(FLANGER_DEPTH_MS * FS / 1000.0)
    max_delay = int(MAX_FLANGER_DELAY_MS * FS / 1000.0)

    base_delay = max(0, min(base_delay, max_delay))
    depth_samples = max(0, min(depth_samples, max_delay - base_delay))

    buf = flanger_buffer
    idx = flanger_idx
    L = len(buf)

    for n in range(len(x)):
        # write current sample into delay line
        buf[idx] = x[n]

        # LFO
        lfo = math.sin(flanger_phase)
        flanger_phase += 2.0 * math.pi * FLANGER_RATE / FS
        lfo_norm = 0.5 * (lfo + 1.0)  # 0..1

        delay = base_delay + int(depth_samples * lfo_norm)
        read_idx = (idx - delay) % L
        delayed = buf[read_idx]

        # mix (dry + 1.0 * wet) - more pronounced flanger effect
        y[n] = x[n] + 1.0 * delayed

        idx += 1
        if idx == L:
            idx = 0

    flanger_idx = idx

    peak = np.max(np.abs(y)) + 1e-9
    if peak > 1.0:
        y *= 1.0 / peak
    return y


# Recording
RECORDING = False
record_buffer = []
record_filename = ""
record_lock = threading.Lock()


def start_recording(filename: str):
    """
    Start recording audio to recordings/<filename>.wav
    """
    global RECORDING, record_buffer, record_filename

    if RECORDING:
        print('[record] Already recording.')
        return

    if not filename:
        filename = 'take.wav'
    if not filename.lower().endswith('.wav'):
        filename += '.wav'

    os.makedirs('recordings', exist_ok=True)
    record_filename = os.path.join('recordings', filename)

    with record_lock:
        record_buffer = []

    RECORDING = True
    print(f'[record] Recording started -> {record_filename}')


def stop_recording():
    """
    Stop recording and write the WAV file.
    """
    global RECORDING, record_buffer, record_filename

    if not RECORDING:
        print('[record] Not currently recording.')
        return

    RECORDING = False

    with record_lock:
        if not record_buffer:
            print('[record] No audio captured, nothing saved.')
            record_filename = ""
            return
        audio = np.concatenate(record_buffer)
        record_buffer = []

    if audio.size == 0:
        print('[record] No audio captured, nothing saved.')
        record_filename = ""
        return

    audio = np.clip(audio, -1.0, 1.0)
    wav_data = (audio * 32767).astype(np.int16)
    wavfile.write(record_filename, FS, wav_data)
    print(f'[record] Saved {record_filename}')
    record_filename = ""


def recording_status():
    """
    Print current recording status.
    """
    if RECORDING:
        with record_lock:
            total_samples = sum(len(chunk) for chunk in record_buffer)
        seconds = total_samples / float(FS)
        print(f'[record] Recording... {seconds:.2f} s so far.')
    else:
        print('[record] Not recording.')


# Terminal-controllable musical scale and mode
CURRENT_SCALE = 'c major'     # updated via terminal
CURRENT_MODE = 'hand'         # 'hand' or 'arm', updated via terminal


def set_global_scale(scale_name):
    """Called from the terminal command thread."""
    global CURRENT_SCALE
    CURRENT_SCALE = scale_name.lower().strip()


def set_global_mode(mode_name):
    """Called from the terminal command thread."""
    global CURRENT_MODE
    if mode_name in ('hand', 'arm'):
        CURRENT_MODE = mode_name


# Instrument/timbre system
INSTRUMENT_PROFILES = {
    'fundamental': np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    'piano': np.array([1.0, 0.75, 0.55, 0.40, 0.28, 0.20, 0.14, 0.10], dtype=np.float32),
    'flute': np.array([1.0, 0.2, 0.10, 0.05, 0.03, 0.02, 0.01, 0.005], dtype=np.float32),
    'clarinet': np.array([1.0, 0.0, 0.7, 0.0, 0.5, 0.0, 0.3, 0.0], dtype=np.float32),
    'organ': np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3], dtype=np.float32),
    'violin': np.array([1.0, 0.95, 1.0, 0.95, 0.85, 0.75, 0.65, 0.55], dtype=np.float32),
}

# person index (1-based), instrument name (string)
PERSON_INSTRUMENTS = {}   # e.g. {1: 'piano', 2: 'flute'}


def list_instruments():
    return list(INSTRUMENT_PROFILES.keys())


def set_person_instrument(person_index, instrument_name):
    """
    Assign an instrument to a given person index (1-based).
    This is called from the terminal command thread.
    """
    global PERSON_INSTRUMENTS
    name = instrument_name.strip().lower()
    if name not in INSTRUMENT_PROFILES:
        print(f'[instrument] Unknown instrument {instrument_name}. '
              f"Valid instruments: {', '.join(list_instruments())}")
        return False
    PERSON_INSTRUMENTS[person_index] = name
    print(f'[instrument] Person {person_index} instrument -> {name}')
    return True


def get_instrument_for_person(person_index):
    """
    Return the instrument name for this person index, defaulting to 'fundamental'.
    """
    name = PERSON_INSTRUMENTS.get(person_index, 'fundamental')
    if name not in INSTRUMENT_PROFILES:
        name = 'fundamental'
    return name


def apply_instrument_profile(period, instrument_name):
    """
    Given a single-period waveform (pose-derived) and an instrument name,
    build an 8-harmonic timbre using that period as the basis waveform.
    """
    period = np.asarray(period, dtype=np.float32)
    L = len(period)
    if L == 0:
        return period

    name = instrument_name.strip().lower()
    weights = INSTRUMENT_PROFILES.get(name, INSTRUMENT_PROFILES['fundamental'])

    base = period - np.mean(period)

    result = np.zeros_like(base, dtype=np.float32)
    idx_base = np.arange(L, dtype=np.int64)

    for h_idx, amp in enumerate(weights):
        if amp == 0.0:
            continue
        k = h_idx + 1
        idxs = (idx_base * k) % L
        result += amp * base[idxs]

    peak = np.max(np.abs(result)) + 1e-6
    result = result / peak * 0.3
    return result.astype(np.float32)


# Pitch and scale utilities
MIN_FREQ = 27.5   # A0
MAX_FREQ = min(4186, FS // 2 - 100.0)  # C8

MIN_CUTOFF = (MIN_FREQ + MAX_FREQ) / 2
MAX_CUTOFF = min(FS // 2 - 100.0, 4000.0)

HP_MIN_CUTOFF = 10.0
HP_MAX_CUTOFF = min(FS // 2 - 100.0, (MIN_FREQ + MAX_FREQ) / 2)

lock = threading.Lock()
stream = None

periods = []
phases = []

cutoffs = []
lp_sos_list = []
lp_zi_list = []

hp_cutoffs = []
hp_sos_list = []
hp_zi_list = []

_NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F',
               'F#', 'G', 'G#', 'A', 'A#', 'B']

_NOTE_TO_PC = {
    'c': 0, 'c#': 1, 'db': 1,
    'd': 2, 'd#': 3, 'eb': 3,
    'e': 4, 'fb': 4, 'e#': 5,
    'f': 5, 'f#': 6, 'gb': 6,
    'g': 7, 'g#': 8, 'ab': 8,
    'a': 9, 'a#': 10, 'bb': 10,
    'b': 11, 'cb': 11, 'b#': 0,
}


def _freq_to_midi(f):
    return 69.0 + 12.0 * math.log2(f / 440.0)


def _midi_to_freq(m):
    return 440.0 * (2.0 ** ((m - 69.0) / 12.0))


def _midi_to_name(m):
    m_int = int(round(m))
    pc = m_int % 12
    octave = m_int // 12 - 1
    return f'{_NOTE_NAMES[pc]}{octave}'


def _parse_scale_name(scale_name):
    s = scale_name.strip().lower().replace(' ', '')
    mode = 'major'
    if 'major' in s:
        mode = 'major'
        root_str = s.replace('major', '')
    elif 'minor' in s:
        mode = 'minor'
        root_str = s.replace('minor', '')
    else:
        root_str = s
        mode = 'major'

    root_str = root_str or 'c'
    root_str = root_str.replace('♯', '#').replace('♭', 'b')

    root_pc = _NOTE_TO_PC.get(root_str, 0)
    return root_pc, mode


def _build_scale_classes(scale_name):
    root_pc, mode = _parse_scale_name(scale_name)

    if mode == 'minor':
        intervals = [0, 2, 3, 5, 7, 8, 10]
    else:
        intervals = [0, 2, 4, 5, 7, 9, 11]

    return [(root_pc + i) % 12 for i in intervals]


def quantize_frequency_to_scale(freq, scale_name):
    freq = float(np.clip(freq, MIN_FREQ, MAX_FREQ))
    allowed = _build_scale_classes(scale_name)

    orig_midi = _freq_to_midi(freq)
    best_m = orig_midi
    best_f = freq
    best_err = float('inf')

    for m in range(0, 128):
        if (m % 12) not in allowed:
            continue
        f = _midi_to_freq(m)
        if f < MIN_FREQ or f > MAX_FREQ:
            continue
        err = abs(m - orig_midi)
        if err < best_err:
            best_err = err
            best_m = m
            best_f = f

    return best_f, _midi_to_name(best_m)


def pose_to_waveform(keypoints):
    # Extract metadata
    if len(keypoints) > 0 and isinstance(keypoints[-1], tuple):
        metadata = keypoints[-1]
        pts = np.array(keypoints[:-1], dtype=float)
    else:
        metadata = None
        pts = np.array(keypoints, dtype=float)

    center = pts[2]
    rel = pts - center
    rel = rel[np.argsort(rel[:, 0])]

    xs = rel[:, 0]
    ys = -rel[:, 1]
    ys = ys / (np.max(np.abs(ys)) + 1e-9)

    jitter = np.random.normal(0, 1e-10, len(xs))
    xs = xs + jitter

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

    # RAW_MIN = 10
    # RAW_MAX = 1000

    # raw_width = float(t[-1])
    # norm = (RAW_MAX - raw_width) / (RAW_MAX - RAW_MIN)
    # norm = np.clip(norm, 0.0, 1.0)

    # gamma = 0.4
    # v = norm ** gamma

    # freq = float(MIN_FREQ * ((MAX_FREQ / MIN_FREQ) ** v))
    # freq = float(np.clip(freq, MIN_FREQ, MAX_FREQ))

    avg_y = None
    avg_x = None
    frame_h = 480
    frame_w = 640

    if metadata is not None and isinstance(metadata, (list, tuple)):
        try:
            if len(metadata) > 0:
                avg_y = float(metadata[0])
            if len(metadata) > 1:
                frame_h = int(metadata[1])
            if len(metadata) > 2:
                avg_x = float(metadata[2])
            if len(metadata) > 3:
                frame_w = int(metadata[3])
        except Exception:
            pass

    # Calculate LAMBDA_MAX based on actual frame diagonal
    LAMBDA_MIN = 10.0
    LAMBDA_MAX = float(np.sqrt(frame_w**2 + frame_h**2)) / 3

    # Gesture size to frequency mapping
    lam = float(np.linalg.norm(pts[-1] - pts[0]))
    lam = float(np.clip(lam, LAMBDA_MIN, LAMBDA_MAX))

    s = np.log2(lam / LAMBDA_MAX) / np.log2(LAMBDA_MIN / LAMBDA_MAX)
    s = float(np.clip(s, 0.0, 1.0))
    freq = float(MIN_FREQ + s * (MAX_FREQ - MIN_FREQ))
    freq = float(np.clip(freq, MIN_FREQ, MAX_FREQ))

    freq_q, note_name = quantize_frequency_to_scale(freq, CURRENT_SCALE)
    # print('lam', lam, f'({LAMBDA_MIN}..{LAMBDA_MAX})', f's = {s}', freq_q, freq, note_name)
    # print(freq_q, note_name, f'(raw freq: {freq:.2f} Hz)', f'gesture size: {lam:.1f}px')

    NUM_BINS = 8

    if avg_y is None:
        lp_cutoff = (MIN_CUTOFF + MAX_CUTOFF) / 2.0
        lp_bin_idx = NUM_BINS // 2
    else:
        frame_h = max(1, frame_h)
        norm_y = np.clip(avg_y / float(frame_h), 0.0, 1.0)
        lp_bin_idx = int(norm_y * NUM_BINS)
        lp_bin_idx = int(np.clip(lp_bin_idx, 0, NUM_BINS - 1))
        lp_bins = np.linspace(MIN_CUTOFF, MAX_CUTOFF, NUM_BINS)
        lp_cutoff = float(lp_bins[lp_bin_idx])

    if avg_x is None:
        hp_cutoff = (HP_MIN_CUTOFF + HP_MAX_CUTOFF) / 2.0
        hp_bin_idx = NUM_BINS // 2
    else:
        frame_w = max(1, frame_w)
        norm_x = np.clip(avg_x / float(frame_w), 0.0, 1.0)
        hp_bin_idx = int(norm_x * NUM_BINS)
        hp_bin_idx = int(np.clip(hp_bin_idx, 0, NUM_BINS - 1))
        hp_bins = np.linspace(HP_MIN_CUTOFF, HP_MAX_CUTOFF, NUM_BINS)
        hp_cutoff = float(hp_bins[hp_bin_idx])

    return wave, freq_q, note_name, lp_cutoff, lp_bin_idx, hp_cutoff, hp_bin_idx


def _wave_to_period(wave, freq):
    freq = float(np.clip(freq, MIN_FREQ, MAX_FREQ))
    ps = max(2, int(FS / freq))
    p = resample(wave, ps).astype(np.float32)
    p /= (np.max(np.abs(p)) + 1e-6)
    p *= 0.3
    return p


def update_audio_from_multiple(wave_freq_cutoff_list):
    global periods, phases, cutoffs, lp_sos_list, lp_zi_list
    global hp_cutoffs, hp_sos_list, hp_zi_list

    new_periods = []
    new_cutoffs = []
    new_sos_list = []
    new_zi_list = []

    new_hp_cutoffs = []
    new_hp_sos_list = []
    new_hp_zi_list = []

    nyq = FS * 0.5

    for idx, (wave, freq, lp_cutoff, hp_cutoff) in enumerate(wave_freq_cutoff_list):
        p_base = _wave_to_period(wave, freq)
        if len(p_base) == 0:
            continue

        person_index = idx + 1
        instr_name = get_instrument_for_person(person_index)
        p = apply_instrument_profile(p_base, instr_name)

        lp_cutoff = float(np.clip(lp_cutoff, MIN_CUTOFF, MAX_CUTOFF))
        lp_Wn = lp_cutoff / nyq

        try:
            lp_sos = signal.butter(4, lp_Wn, btype='low', output='sos')
            lp_zi = signal.sosfilt_zi(lp_sos) * 0.0
        except Exception as ex:
            print(f'[update_audio_from_multiple] LP filter design failed: {ex}')
            lp_sos = None
            lp_zi = None

        hp_cutoff = float(np.clip(hp_cutoff, HP_MIN_CUTOFF, HP_MAX_CUTOFF))
        hp_Wn = hp_cutoff / nyq

        try:
            hp_sos = signal.butter(4, hp_Wn, btype='high', output='sos')
            hp_zi = signal.sosfilt_zi(hp_sos) * 0.0
        except Exception as ex:
            print(f'[update_audio_from_multiple] HP filter design failed: {ex}')
            hp_sos = None
            hp_zi = None

        new_periods.append(p)
        new_cutoffs.append(lp_cutoff)
        new_sos_list.append(lp_sos)
        new_zi_list.append(lp_zi)

        new_hp_cutoffs.append(hp_cutoff)
        new_hp_sos_list.append(hp_sos)
        new_hp_zi_list.append(hp_zi)

    with lock:
        periods = new_periods
        
        # Preserve existing phases for continuity, only initialize new voices
        if len(new_periods) > len(phases):
            phases.extend([0] * (len(new_periods) - len(phases)))
        else:
            phases = phases[:len(new_periods)]

        cutoffs = new_cutoffs
        
        # Preserve existing LP filter states for continuity
        # Only create new states for newly added voices
        num_old_lp = len(lp_sos_list)
        num_new_lp = len(new_sos_list)
        
        if num_new_lp > num_old_lp:
            # More voices - keep old states, add new ones
            lp_sos_list = lp_sos_list + new_sos_list[num_old_lp:]
            lp_zi_list = lp_zi_list + new_zi_list[num_old_lp:]
        else:
            # Same or fewer voices
            lp_sos_list = lp_sos_list[:num_new_lp]
            lp_zi_list = lp_zi_list[:num_new_lp]
        
        # Update the sos coefficients for all voices (in case cutoff changed)
        for i in range(min(num_old_lp, num_new_lp)):
            lp_sos_list[i] = new_sos_list[i]
        
        # Same for HP filters
        num_old_hp = len(hp_sos_list)
        num_new_hp = len(new_hp_sos_list)
        
        if num_new_hp > num_old_hp:
            hp_sos_list = hp_sos_list + new_hp_sos_list[num_old_hp:]
            hp_zi_list = hp_zi_list + new_hp_zi_list[num_old_hp:]
        else:
            hp_sos_list = hp_sos_list[:num_new_hp]
            hp_zi_list = hp_zi_list[:num_new_hp]
        
        for i in range(min(num_old_hp, num_new_hp)):
            hp_sos_list[i] = new_hp_sos_list[i]

        hp_cutoffs = new_hp_cutoffs


def update_audio_from_pose(keypoints):
    wave, freq, note_name, lp_cutoff, lp_bin_idx, hp_cutoff, hp_bin_idx = pose_to_waveform(keypoints)
    update_audio_from_multiple([(wave, freq, lp_cutoff, hp_cutoff)])


def audio_callback(outdata, frames, time, status):
    global periods, phases, lp_sos_list, lp_zi_list, hp_sos_list, hp_zi_list

    if status:
        print(f'[audio] Status: {status}')

    try:
        # Copy DSP state snapshot
        with lock:
            local_periods = list(periods)
            local_phases = phases.copy()
            local_lp_sos_list = list(lp_sos_list)
            local_lp_zi_list = [zi.copy() if zi is not None else None for zi in lp_zi_list]
            local_hp_sos_list = list(hp_sos_list)
            local_hp_zi_list = [zi.copy() if zi is not None else None for zi in hp_zi_list]

        nvoices = len(local_periods)

        if not (
            len(local_phases) == len(local_lp_sos_list) ==
            len(local_lp_zi_list) == len(local_hp_sos_list) ==
            len(local_hp_zi_list) == nvoices
        ):
            outdata[:] = 0.0
            return

        # No input case
        if nvoices == 0:
            # Pedal on
            if PEDAL_MODE:
                dry = np.zeros(frames, dtype=np.float32)
                out = process_reverb_block(dry)
                if FLANGER_ON:
                    out = apply_flanger_block(out)
            else:
                out = np.zeros(frames, dtype=np.float32)

            # Recording
            if RECORDING:
                with record_lock:
                    record_buffer.append(out.copy())

            if outdata.ndim == 1:
                outdata[:] = out
            else:
                for ch in range(outdata.shape[1]):
                    outdata[:, ch] = out
            return

        # At least one voice case
        out = np.zeros(frames, dtype=np.float32)

        for v in range(nvoices):
            try:
                p = np.asarray(local_periods[v], dtype=np.float32)
                L = len(p)
                if L == 0:
                    continue

                phase = int(local_phases[v])
                idxs = (np.arange(frames) + phase) % L
                voice = p[idxs]

                # Highpass
                hp_sos = local_hp_sos_list[v]
                hp_zi = local_hp_zi_list[v]
                if hp_sos is not None and hp_zi is not None:
                    voice, local_hp_zi_list[v] = signal.sosfilt(hp_sos, voice, zi=hp_zi)

                # Lowpass
                lp_sos = local_lp_sos_list[v]
                lp_zi = local_lp_zi_list[v]
                if lp_sos is not None and lp_zi is not None:
                    voice, local_lp_zi_list[v] = signal.sosfilt(lp_sos, voice, zi=lp_zi)

                out += voice
                local_phases[v] = int((phase + frames) % L)

            except Exception as ex:
                print(f'[audio_callback] voice {v} skipped: {ex}')
                continue

        if nvoices > 1:
            out /= np.sqrt(nvoices)

        # Normalize
        peak = np.max(np.abs(out)) + 1e-9
        if peak > 0.3:
            out *= (0.3 / peak)

        # Pedal then flanger
        out = process_reverb_block(out)
        if FLANGER_ON:
            out = apply_flanger_block(out)

        # Recording
        if RECORDING:
            with record_lock:
                record_buffer.append(out.copy())

        # Output audio
        if outdata.ndim == 1:
            outdata[:] = out
        else:
            for ch in range(outdata.shape[1]):
                outdata[:, ch] = out

        # Write back state
        with lock:
            if (
                len(phases) == nvoices and
                len(lp_zi_list) == nvoices and
                len(hp_zi_list) == nvoices
            ):
                phases = local_phases
                lp_zi_list = local_lp_zi_list
                hp_zi_list = local_hp_zi_list

    except Exception as e:
        print(f'[audio_callback] Error: {e}')
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
