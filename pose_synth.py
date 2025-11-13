# Retry: create pose_synth.py and README.md as files in /mnt/data

from textwrap import dedent

pose_synth_code = dedent(r"""#!/usr/bin/env python3
# Real-Time Arm Pose Classification for Audio Synthesis
# ----------------------------------------------------
#
# Dependencies (install via pip):
#     pip install mediapipe opencv-python numpy sounddevice scipy
#
# Optional (for better portability on some systems):
#     pip install cffi
#
# Run:
#     python pose_synth.py
#
# Keys:
#     q               Quit
#     f/F             -/+ base frequency (Hz)
#     1/2/3/4         Force waveform: 1=sine, 2=square, 3=saw, 4=triangle (toggle "auto" off)
#     0               Return to AUTO mapping from pose
#     e               Toggle EQ (filter enable)
#     d               Toggle distortion
#     r               Toggle reverb
#     a               Toggle harmonics auto-count VS manual count
#     h/H             -/+ number of harmonics (manual mode)
#     m               Mute toggle
#
# OpenCV Sliders (top window):
#     drive           Distortion drive
#     lp_cut          Low-pass cutoff (Hz)
#     bp_c_low        Band-pass low (Hz)
#     bp_c_high       Band-pass high (Hz)
#     hp_cut          High-pass cutoff (Hz)
#     eq_mix          EQ wet/dry (0..100)
#     rev_mix         Reverb wet/dry (0..100)
#     out_gain        Output gain (dB, -24..+24)
#
# Notes:
#     - This is a single-file reference implementation for a class project. It's designed
#       for clarity and hackability over maximum performance.
#     - Ensure your default input camera is available. Allow camera permissions if prompted.
#     - Audio latency is sensitive to your device/OS and sounddevice backend.
#
import math
import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import sounddevice as sd
from scipy import signal

try:
    import mediapipe as mp
except Exception as e:
    raise SystemExit(
        "Failed to import mediapipe. Install it with: pip install mediapipe\n"
        f"Original error: {e}"
    )

# ---------------------------- Config ---------------------------------

SAMPLE_RATE = 48000
BLOCK_SIZE = 512
DEFAULT_FREQ = 220.0  # A3
MAX_HARMONICS = 64

# ---------------------- Pose & Waveform Mapping ----------------------

mp_pose = mp.solutions.pose

@dataclass
class PoseFeatures:
    left_elbow_deg: float
    right_elbow_deg: float
    left_arm_slope: float
    right_arm_slope: float
    arms_v_symmetry: float
    valid: bool

def angle_between(p0, p1, p2) -> Optional[float]:
    if any(v is None for v in (p0, p1, p2)):
        return None
    a = np.array([p0[0] - p1[0], p0[1] - p1[1]], dtype=float)
    b = np.array([p2[0] - p1[0], p2[1] - p1[1]], dtype=float)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-6 or nb < 1e-6:
        return None
    cosang = np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))

def line_slope(p0, p1) -> Optional[float]:
    if any(v is None for v in (p0, p1)):
        return None
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    if abs(dx) < 1e-6:
        return None
    return float(dy / dx)

def extract_pose_features(landmarks, image_w, image_h) -> PoseFeatures:
    def get_xy(idx):
        lm = landmarks[idx]
        return (lm.x * image_w, lm.y * image_h)

    L_SHO, R_SHO = 11, 12
    L_ELB, R_ELB = 13, 14
    L_WRI, R_WRI = 15, 16

    try:
        ls, rs = get_xy(L_SHO), get_xy(R_SHO)
        le, re = get_xy(L_ELB), get_xy(R_ELB)
        lw, rw = get_xy(L_WRI), get_xy(R_WRI)
    except Exception:
        return PoseFeatures(0, 0, 0, 0, 0, False)

    left_elbow = angle_between(ls, le, lw)
    right_elbow = angle_between(rs, re, rw)
    left_slope = line_slope(le, lw)
    right_slope = line_slope(re, rw)

    v_sym = 0.0
    if None not in (left_slope, right_slope, le[1], lw[1], re[1], rw[1]):
        left_up = lw[1] < le[1]
        right_up = rw[1] < re[1]
        slope_sym = 1.0 - min(1.0, abs((left_slope or 0) + (right_slope or 0)) / 2.0)
        v_sym = float(left_up and right_up) * slope_sym

    valid = all(v is not None for v in (left_elbow, right_elbow, left_slope, right_slope))
    return PoseFeatures(
        left_elbow_deg=left_elbow or 0.0,
        right_elbow_deg=right_elbow or 0.0,
        left_arm_slope=left_slope or 0.0,
        right_arm_slope=right_slope or 0.0,
        arms_v_symmetry=v_sym,
        valid=bool(valid),
    )

class PoseToWaveMapper:
    WAVE_SINE = "sine"
    WAVE_SQUARE = "square"
    WAVE_SAW = "saw"
    WAVE_TRIANGLE = "triangle"

    def __init__(self):
        self.forced: Optional[str] = None

    def set_forced(self, name: Optional[str]):
        self.forced = name

    @staticmethod
    def _score_sine(f: PoseFeatures) -> float:
        left = 1.0 - min(1.0, abs(180.0 - f.left_elbow_deg) / 30.0)
        right = 1.0 - min(1.0, abs(180.0 - f.right_elbow_deg) / 30.0)
        return 0.5 * (left + right)

    @staticmethod
    def _score_square(f: PoseFeatures) -> float:
        left = 1.0 - min(1.0, abs(90.0 - f.left_elbow_deg) / 30.0)
        right = 1.0 - min(1.0, abs(90.0 - f.right_elbow_deg) / 30.0)
        return 0.5 * (left + right)

    @staticmethod
    def _score_triangle(f: PoseFeatures) -> float:
        return float(f.arms_v_symmetry)

    @staticmethod
    def _score_saw(f: PoseFeatures) -> float:
        s_left = 1.0 - min(1.0, abs(abs(f.left_arm_slope) - 1.0) / 1.0)
        s_right = 1.0 - min(1.0, abs(abs(f.right_arm_slope) - 1.0) / 1.0)
        return max(s_left, s_right)

    def choose(self, f: PoseFeatures) -> str:
        if self.forced:
            return self.forced
        if not f.valid:
            return self.WAVE_SINE

        scores = {
            self.WAVE_SINE: self._score_sine(f),
            self.WAVE_SQUARE: self._score_square(f),
            self.WAVE_SAW: self._score_saw(f),
            self.WAVE_TRIANGLE: self._score_triangle(f),
        }
        vals = np.array(list(scores.values()), dtype=float)
        expv = np.exp(vals * 3.0)
        probs = expv / np.sum(expv + 1e-8)
        choice = list(scores.keys())[int(np.argmax(probs))]
        return choice

# ---------------------------- Synth Engine ---------------------------

@dataclass
class SynthParams:
    freq: float = DEFAULT_FREQ
    waveform: str = "sine"
    harmonics_auto: bool = True
    harmonics: int = 16
    eq_enabled: bool = True
    eq_lp_cut: float = 8000.0
    eq_bp_low: float = 500.0
    eq_bp_high: float = 2000.0
    eq_mix: float = 0.5
    distortion_enabled: bool = False
    drive: float = 1.0
    reverb_enabled: bool = False
    reverb_mix: float = 0.2
    mute: bool = False
    out_gain_db: float = 0.0

class SimpleReverb:
    def __init__(self, sr=SAMPLE_RATE):
        self.comb_delays = [int(sr * t) for t in (0.0297, 0.0371, 0.0411)]
        self.comb_buffers = [np.zeros(d, dtype=np.float32) for d in self.comb_delays]
        self.comb_idx = [0] * len(self.comb_delays)
        self.comb_feedback = [0.77, 0.74, 0.73]

        ap_delay = int(sr * 0.005)
        self.ap_buf = np.zeros(ap_delay, dtype=np.float32)
        self.ap_idx = 0
        self.ap_g = 0.5

    def process(self, x: np.ndarray) -> np.ndarray:
        y = np.zeros_like(x)
        for i, buf in enumerate(self.comb_buffers):
            idx = self.comb_idx[i]
            fb = self.comb_feedback[i]
            out = np.zeros_like(x)
            for n in range(x.shape[0]):
                v = buf[idx]
                out[n] = v
                buf[idx] = x[n] + fb * v
                idx += 1
                if idx >= buf.size:
                    idx = 0
            self.comb_idx[i] = idx
            y += out / len(self.comb_buffers)

        out_ap = np.zeros_like(x)
        for n in range(x.shape[0]):
            v = x[n] + y[n]
            z = self.ap_buf[self.ap_idx]
            out_ap[n] = -self.ap_g * v + z + self.ap_g * (v)
            self.ap_buf[self.ap_idx] = v + self.ap_g * z
            self.ap_idx += 1
            if self.ap_idx >= self.ap_buf.size:
                self.ap_idx = 0
        return out_ap

class SynthEngine:
    def __init__(self, params: SynthParams, sr: int = SAMPLE_RATE):
        self.params = params
        self.sr = sr
        self.phase = 0.0
        self.lock = threading.Lock()

        self._sos_lp = None
        self._sos_bp = None
        self._sos_hp = None
        self._zi_lp = None
        self._zi_bp = None
        self._zi_hp = None

        self.reverb = SimpleReverb(sr=sr)

    def _ensure_filters(self):
        nyq = 0.5 * self.sr
        try:
            self._sos_lp = signal.iirfilter(4, self.params.eq_lp_cut / nyq, btype='low', ftype='butter', output='sos')
            self._zi_lp = signal.sosfilt_zi(self._sos_lp)
        except Exception:
            self._sos_lp = None
            self._zi_lp = None

        try:
            low = max(5.0, self.params.eq_bp_low) / nyq
            high = min(nyq - 100, self.params.eq_bp_high) / nyq
            high = max(high, low + 1e-3)
            self._sos_bp = signal.iirfilter(4, [low, high], btype='band', ftype='butter', output='sos')
            self._zi_bp = signal.sosfilt_zi(self._sos_bp)
        except Exception:
            self._sos_bp = None
            self._zi_bp = None

        try:
            self._sos_hp = signal.iirfilter(4, max(5.0, self.params.eq_hp_cut) / nyq, btype='high', ftype='butter', output='sos')
            self._zi_hp = signal.sosfilt_zi(self._sos_hp)
        except Exception:
            self._sos_hp = None
            self._zi_hp = None

    def set_params(self, **kwargs):
        with self.lock:
            old_lp = self.params.eq_lp_cut
            old_bp = (self.params.eq_bp_low, self.params.eq_bp_high)
            old_hp = self.params.eq_hp_cut

            for k, v in kwargs.items():
                if hasattr(self.params, k):
                    setattr(self.params, k, v)

            if (
                self.params.eq_lp_cut != old_lp or
                (self.params.eq_bp_low, self.params.eq_bp_high) != old_bp or
                self.params.eq_hp_cut != old_hp
            ):
                self._ensure_filters()

    @staticmethod
    def _fourier_series(waveform: str, harmonics: int, t: np.ndarray, freq: float, sr: int) -> np.ndarray:
        w = 2.0 * np.pi * freq
        out = np.zeros_like(t, dtype=np.float32)
        if waveform == "sine":
            out = np.sin(w * t, dtype=np.float32)
        elif waveform == "square":
            out_f = np.zeros_like(t, dtype=np.float32)
            for k in range(1, 2*harmonics, 2):
                out_f += np.sin(k * w * t) / k
            out = (4/np.pi) * out_f
        elif waveform == "saw":
            out_f = np.zeros_like(t, dtype=np.float32)
            for k in range(1, harmonics+1):
                sign = 1.0 if (k % 2 == 1) else -1.0
                out_f += sign * (np.sin(k * w * t) / k)
            out = (2/np.pi) * out_f
        elif waveform == "triangle":
            out_f = np.zeros_like(t, dtype=np.float32)
            sign = 1.0
            for k in range(1, 2*harmonics, 2):
                out_f += sign * (np.sin(k * w * t) / (k*k))
                sign *= -1.0
            out = (8/(np.pi**2)) * out_f
        else:
            out = np.sin(w * t, dtype=np.float32)
        maxv = np.max(np.abs(out)) + 1e-9
        out = (out / maxv).astype(np.float32)
        return out

    def render(self, frames: int) -> np.ndarray:
        with self.lock:
            p = self.params

        if p.mute:
            return np.zeros(frames, dtype=np.float32)

        if p.harmonics_auto:
            max_h = int((0.45 * self.sr) // max(1.0, p.freq))
            harmonics = max(1, min(MAX_HARMONICS, max_h))
        else:
            harmonics = max(1, min(MAX_HARMONICS, p.harmonics))

        t = (np.arange(frames, dtype=np.float32) + self.phase) / self.sr
        self.phase = (self.phase + frames) % self.sr

        x = self._fourier_series(p.waveform, harmonics, t, p.freq, self.sr)

        if p.distortion_enabled:
            drive = max(0.1, float(p.drive))
            x = np.tanh(drive * x).astype(np.float32)

        if p.eq_enabled:
            wet = x.copy()
            if self._sos_lp is None or self._sos_bp is None or self._sos_hp is None:
                self._ensure_filters()
            if self._sos_lp is not None:
                wet, self._zi_lp = signal.sosfilt(self._sos_lp, wet, zi=self._zi_lp)
            if self._sos_bp is not None:
                wet, self._zi_bp = signal.sosfilt(self._sos_bp, wet, zi=self._zi_bp)
            if self._sos_hp is not None:
                wet, self._zi_hp = signal.sosfilt(self._sos_hp, wet, zi=self._zi_hp)
            x = (1.0 - p.eq_mix) * x + p.eq_mix * wet

        if p.reverb_enabled:
            wet = self.reverb.process(x.astype(np.float32))
            x = (1.0 - p.reverb_mix) * x + p.reverb_mix * wet

        gain = 10 ** (p.out_gain_db / 20.0)
        x = np.clip(x * gain, -1.0, 1.0).astype(np.float32)
        return x

# ------------------------------ App ----------------------------------

class SharedState:
    def __init__(self):
        self.params = SynthParams()
        self.mapper = PoseToWaveMapper()
        self.last_waveform = "sine"
        self.last_spectrum = np.zeros(256, dtype=np.float32)
        self.lock = threading.Lock()

def audio_callback(outdata, frames, time_info, status, engine: SynthEngine):
    if status:
        pass
    out = engine.render(frames)
    outdata[:, 0] = out
    if outdata.shape[1] > 1:
        outdata[:, 1] = out

def create_trackbars(win: str, state: SharedState):
    def nothing(_): pass
    cv2.createTrackbar("drive", win, 10, 100, nothing)
    cv2.createTrackbar("lp_cut", win, 8000, 20000, nothing)
    cv2.createTrackbar("bp_c_low", win, 500, 8000, nothing)
    cv2.createTrackbar("bp_c_high", win, 2000, 20000, nothing)
    cv2.createTrackbar("hp_cut", win, 80, 2000, nothing)
    cv2.createTrackbar("eq_mix", win, 50, 100, nothing)
    cv2.createTrackbar("rev_mix", win, 20, 100, nothing)
    cv2.createTrackbar("out_gain", win, 24, 48, nothing)

def read_trackbars(win: str, state: SharedState, engine: SynthEngine):
    drive = max(1, cv2.getTrackbarPos("drive", win)) / 10.0
    lp_cut = max(50, cv2.getTrackbarPos("lp_cut", win))
    bp_low = max(10, cv2.getTrackbarPos("bp_c_low", win))
    bp_high = max(bp_low + 10, cv2.getTrackbarPos("bp_c_high", win))
    hp_cut = max(10, cv2.getTrackbarPos("hp_cut", win))
    eq_mix = cv2.getTrackbarPos("eq_mix", win) / 100.0
    rev_mix = cv2.getTrackbarPos("rev_mix", win) / 100.0
    out_gain = cv2.getTrackbarPos("out_gain", win) - 24

    engine.set_params(
        drive=drive,
        eq_lp_cut=float(lp_cut),
        eq_bp_low=float(bp_low),
        eq_bp_high=float(bp_high),
        eq_hp_cut=float(hp_cut),
        eq_mix=float(eq_mix),
        reverb_mix=float(rev_mix),
        out_gain_db=float(out_gain),
    )

def draw_pose_and_spectrum(frame, pose_results, waveform_name: str, freq: float, spectrum: np.ndarray):
    h, w = frame.shape[:2]
    if pose_results and pose_results.pose_landmarks:
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing.draw_landmarks(
            frame,
            pose_results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 128, 0), thickness=2),
        )

    cv2.rectangle(frame, (0, 0), (w, 80), (0, 0, 0), -1)
    cv2.putText(frame, f"Wave: {waveform_name}  Freq: {freq:.1f} Hz",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(frame, "Keys: [0]AUTO [1]SINE [2]SQUARE [3]SAW [4]TRI  f/F -/+Hz  e EQ  d Dist  r Rev  a HarmAuto  h/H HarmN  m Mute  q Quit",
                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    spec = spectrum
    spec_img_w = 400
    spec_img_h = 120
    spec_img = np.zeros((spec_img_h, spec_img_w, 3), dtype=np.uint8)
    if spec.size > 4:
        s = spec[:spec_img_w]
        s = s / (np.max(s) + 1e-9)
        pts = []
        for i in range(s.size):
            x = i
            y = int((1.0 - s[i]) * (spec_img_h - 1))
            pts.append((x, y))
        for i in range(1, len(pts)):
            cv2.line(spec_img, pts[i-1], pts[i], (255, 255, 255), 1)
    frame[10:10+spec_img_h, w-spec_img_w-10:w-10] = spec_img

def compute_spectrum(samples: np.ndarray, sr: int) -> np.ndarray:
    if samples.size < 8:
        return np.zeros(8, dtype=np.float32)
    win = np.hanning(samples.size)
    X = np.fft.rfft(samples * win)
    mag = np.abs(X).astype(np.float32)
    return mag

def main():
    state = SharedState()
    engine = SynthEngine(state.params, sr=SAMPLE_RATE)

    stream = sd.OutputStream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        channels=2,
        dtype='float32',
        callback=lambda outdata, frames, time_info, status: audio_callback(outdata, frames, time_info, status, engine),
    )
    stream.start()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Could not open default camera. Make sure a webcam is connected and accessible.")

    cv2.namedWindow("PoseSynth", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("PoseSynth", 1280, 720)
    create_trackbars("PoseSynth", state)

    with mp_pose.Pose(model_complexity=0, enable_segmentation=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        vis_buf = np.zeros(BLOCK_SIZE * 8, dtype=np.float32)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            h, w = frame.shape[:2]
            features = None
            if results.pose_landmarks:
                features = extract_pose_features(results.pose_landmarks.landmark, w, h)

            if features:
                wf = state.mapper.choose(features)
            else:
                wf = state.last_waveform

            if state.mapper.forced is not None:
                wf = state.mapper.forced

            engine.set_params(waveform=wf)

            read_trackbars("PoseSynth", state, engine)

            with engine.lock:
                p = engine.params

            frames_vis = 2048
            t = np.arange(frames_vis, dtype=np.float32) / engine.sr
            if p.harmonics_auto:
                max_h = int((0.45 * engine.sr) // max(1.0, p.freq))
                harmonics = max(1, min(MAX_HARMONICS, max_h))
            else:
                harmonics = max(1, min(MAX_HARMONICS, p.harmonics))
            x_vis = engine._fourier_series(p.waveform, harmonics, t, p.freq, engine.sr)
            if p.distortion_enabled:
                x_vis = np.tanh(p.drive * x_vis).astype(np.float32)
            if p.eq_enabled:
                nyq = 0.5 * engine.sr
                try:
                    sos_lp = signal.iirfilter(4, p.eq_lp_cut / nyq, btype='low', ftype='butter', output='sos')
                    x_vis = signal.sosfilt(sos_lp, x_vis)
                except Exception:
                    pass
                try:
                    low = max(5.0, p.eq_bp_low) / nyq
                    high = min(nyq - 100, p.eq_bp_high) / nyq
                    high = max(high, low + 1e-3)
                    sos_bp = signal.iirfilter(4, [low, high], btype='band', ftype='butter', output='sos')
                    x_vis = signal.sosfilt(sos_bp, x_vis)
                except Exception:
                    pass
                try:
                    sos_hp = signal.iirfilter(4, max(5.0, p.eq_hp_cut) / nyq, btype='high', ftype='butter', output='sos')
                    x_vis = signal.sosfilt(sos_hp, x_vis)
                except Exception:
                    pass
            if p.reverb_enabled:
                rv = SimpleReverb(sr=engine.sr)
                x_vis = (1.0 - p.reverb_mix) * x_vis + p.reverb_mix * rv.process(x_vis.astype(np.float32))
            x_vis = np.clip((10 ** (p.out_gain_db / 20.0)) * x_vis, -1.0, 1.0).astype(np.float32)

            vis_buf = np.roll(vis_buf, -x_vis.size)
            vis_buf[-x_vis.size:] = x_vis
            spec = compute_spectrum(vis_buf, engine.sr)
            state.last_spectrum = spec

            draw_pose_and_spectrum(frame, results, wf, p.freq, state.last_spectrum)
            cv2.imshow("PoseSynth", frame)

            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            elif k == ord('f'):
                engine.set_params(freq=max(20.0, p.freq - 5.0))
            elif k == ord('F'):
                engine.set_params(freq=min(2000.0, p.freq + 5.0))
            elif k == ord('1'):
                state.mapper.set_forced(PoseToWaveMapper.WAVE_SINE)
            elif k == ord('2'):
                state.mapper.set_forced(PoseToWaveMapper.WAVE_SQUARE)
            elif k == ord('3'):
                state.mapper.set_forced(PoseToWaveMapper.WAVE_SAW)
            elif k == ord('4'):
                state.mapper.set_forced(PoseToWaveMapper.WAVE_TRIANGLE)
            elif k == ord('0'):
                state.mapper.set_forced(None)
            elif k == ord('e'):
                engine.set_params(eq_enabled=not p.eq_enabled)
            elif k == ord('d'):
                engine.set_params(distortion_enabled=not p.distortion_enabled)
            elif k == ord('r'):
                engine.set_params(reverb_enabled=not p.reverb_enabled)
            elif k == ord('a'):
                engine.set_params(harmonics_auto=not p.harmonics_auto)
            elif k == ord('h'):
                engine.set_params(harmonics=max(1, p.harmonics - 1))
            elif k == ord('H'):
                engine.set_params(harmonics=min(MAX_HARMONICS, p.harmonics + 1))
            elif k == ord('m'):
                engine.set_params(mute=not p.mute)

            state.last_waveform = wf

    stream.stop()
    stream.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
""")

readme = dedent(r"""# Real-Time Arm Pose Classification for Audio Synthesis

This project maps arm poses (webcam) to waveforms and synthesizes sound with Fourier series, plus EQ, distortion, and reverb. A live overlay shows pose and spectrum.

## Install
```bash
python -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate
pip install mediapipe opencv-python numpy sounddevice scipy
# If audio backend issues:
pip install cffi
