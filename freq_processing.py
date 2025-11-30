import numpy as np
import sounddevice as sd
from scipy.signal import resample
from scipy.interpolate import interp1d
from scipy import signal
import threading
import math

FS = 44100

# === REVERB ADDED ===
REVERB_ON = False

def apply_reverb(x, sample_rate=44100, delay_ms=1000, feedback=0.9, mix=0.9):
    """
    Longer and smoother Schroeder-style reverb.
    delay_ms controls the sense of space.
    """
    delay_samples = int(sample_rate * delay_ms / 1000)
    if delay_samples <= 0:
        return x

    y = np.copy(x).astype(np.float32)

    # Simple feedback delay
    if delay_samples < len(y):
        for i in range(delay_samples, len(y)):
            y[i] += feedback * y[i - delay_samples]

    # Normalize to prevent runaway volume
    y /= (np.max(np.abs(y)) + 1e-6)

    # Dry/wet mix
    return (1 - mix) * x + mix * y

# === END REVERB ===


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
MIN_FREQ = 65.41   # C2
MAX_FREQ = min(1046.5, FS // 2 - 100.0)  # C6

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

    RAW_MIN = 10
    RAW_MAX = 1000

    raw_width = float(t[-1])
    norm = (RAW_MAX - raw_width) / (RAW_MAX - RAW_MIN)
    norm = np.clip(norm, 0.0, 1.0)

    gamma = 0.4
    v = norm ** gamma

    freq = float(MIN_FREQ * ((MAX_FREQ / MIN_FREQ) ** v))
    freq = float(np.clip(freq, MIN_FREQ, MAX_FREQ))

    freq_q, note_name = quantize_frequency_to_scale(freq, CURRENT_SCALE)

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
    ps = max(32, int(FS / freq))
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
        phases = [0] * len(new_periods)

        cutoffs = new_cutoffs
        lp_sos_list = new_sos_list
        lp_zi_list = new_zi_list

        hp_cutoffs = new_hp_cutoffs
        hp_sos_list = new_hp_sos_list
        hp_zi_list = new_hp_zi_list


def update_audio_from_pose(keypoints):
    wave, freq, note_name, lp_cutoff, lp_bin_idx, hp_cutoff, hp_bin_idx = pose_to_waveform(keypoints)
    update_audio_from_multiple([(wave, freq, lp_cutoff, hp_cutoff)])


def audio_callback(outdata, frames, time, status):
    global periods, phases, lp_sos_list, lp_zi_list, hp_sos_list, hp_zi_list
    global LAST_REVERB_TAIL

    if status:
        print(f'[audio] Status: {status}')

    try:
        # Copy current DSP state
        with lock:
            local_periods = list(periods)
            local_phases = phases.copy()
            local_lp_sos_list = list(lp_sos_list)
            local_lp_zi_list = [zi.copy() if zi is not None else None for zi in lp_zi_list]
            local_hp_sos_list = list(hp_sos_list)
            local_hp_zi_list = [zi.copy() if zi is not None else None for zi in hp_zi_list]

        nvoices = len(local_periods)

        # ============================================
        #       FIXED: NO INPUT → REVERB TAIL ONLY
        # ============================================
        if nvoices == 0:
            if REVERB_ON:
                # Use previous tail or initialize
                if LAST_REVERB_TAIL is None:
                    LAST_REVERB_TAIL = np.zeros(frames, dtype=np.float32)

                # Feed zero input into reverb to get pure decay
                silence = np.zeros(frames, dtype=np.float32)
                new_tail = apply_reverb(silence)

                # Blend previous tail to create natural decay
                tail = 0.85 * LAST_REVERB_TAIL + 0.15 * new_tail

                # Output the tail
                if outdata.ndim == 1:
                    outdata[:] = tail
                else:
                    for ch in range(outdata.shape[1]):
                        outdata[:, ch] = tail

                # Save for next iteration
                LAST_REVERB_TAIL = tail.copy()
                return

            # Reverb OFF → full silence
            outdata[:] = 0.0
            LAST_REVERB_TAIL = None
            return

        # ============================================
        #      NORMAL CASE: A PERSON IS DETECTED
        # ============================================

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

        # === Apply reverb (wet only to LAST_REVERB_TAIL) ===
        if REVERB_ON:
            out = apply_reverb(out)
            LAST_REVERB_TAIL = out.copy()
        else:
            LAST_REVERB_TAIL = None

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
