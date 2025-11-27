import numpy as np
from scipy import signal
import sounddevice as sd


class Synthesizer:
    def __init__(self, sample_rate=48000):
        self.sample_rate = sample_rate


    def additive_synth_helper(self, freqs, amps, phases, duration):
        t = np.linspace(0, duration, int(self.sample_rate * duration), endpoint=False)
        waveform = np.zeros_like(t)

        for f, a, p in zip(freqs, amps, phases):
            waveform += a * np.sin(2 * np.pi * f * t + p)

        # sanitize and normalize output to avoid numerical issues
        waveform = self._sanitize(waveform, normalize=True)
        return waveform


    def additive_synth(self, X, original_length, duration):
        freqs = np.fft.rfftfreq(original_length, d=1 / self.sample_rate)

        amps = (2.0 / original_length) * np.abs(X)
        amps[0] /= 2.0 
        if original_length % 2 == 0:
            amps[-1] /= 2.0

        phases = np.angle(X)

        return self.additive_synth_helper(freqs, amps, phases, duration)


    def _sanitize(self, data, normalize=True):
        # Convert to numpy array with safe precision for processing
        arr = np.asarray(data, dtype=np.float64)
        # Replace NaN/inf with zeros
        if not np.isfinite(arr).all():
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        if normalize:
            peak = np.max(np.abs(arr)) + 1e-12
            if peak > 0:
                arr = arr / peak

        return arr.astype(np.float32)


    def lowpass_filter(self, data, cutoff_hz, order=4):
        nyq = 0.5 * self.sample_rate
        # Clamp cutoff to a safe range (must be >0 and < Nyquist)
        cutoff_hz = float(np.clip(cutoff_hz, 1.0, max(1.0, nyq - 1.0)))
        # Use second-order sections for numerical stability with higher orders
        try:
            sos = signal.butter(order, cutoff_hz / nyq, btype='low', output='sos')
            filtered = signal.sosfilt(sos, data)
        except Exception:
            # Fallback to transfer-function form if SOS fails for some reason
            b, a = signal.butter(order, cutoff_hz / nyq, btype='low')
            filtered = signal.lfilter(b, a, data)

        # Preserve original peak level: sanitize without normalizing, then scale
        in_peak = np.max(np.abs(data)) + 1e-12
        out = self._sanitize(filtered, normalize=False)
        out_peak = np.max(np.abs(out)) + 1e-12
        if out_peak > 0:
            out = out * (in_peak / out_peak)
        return out.astype(np.float32)
    

    def highpass_filter(self, data, cutoff_hz, order=4):
        nyq = 0.5 * self.sample_rate
        # Clamp cutoff to safe range
        cutoff_hz = float(np.clip(cutoff_hz, 1.0, max(1.0, nyq - 1.0)))
        try:
            sos = signal.butter(order, cutoff_hz / nyq, btype='high', output='sos')
            filtered = signal.sosfilt(sos, data)
        except Exception:
            b, a = signal.butter(order, cutoff_hz / nyq, btype='high')
            filtered = signal.lfilter(b, a, data)

        # Preserve input peak level after filtering
        in_peak = np.max(np.abs(data)) + 1e-12
        out = self._sanitize(filtered, normalize=False)
        out_peak = np.max(np.abs(out)) + 1e-12
        if out_peak > 0:
            out = out * (in_peak / out_peak)
        return out.astype(np.float32)


    def bandpass_filter(self, data, low, high, order=4):
        nyq = 0.5 * self.sample_rate
        # Clamp low/high to safe ranges and ensure low < high
        low = float(np.clip(low, 1.0, max(1.0, nyq - 2.0)))
        high = float(np.clip(high, low + 1.0, max(low + 1.0, nyq - 1.0)))
        try:
            sos = signal.butter(order, [low / nyq, high / nyq], btype='band', output='sos')
            filtered = signal.sosfilt(sos, data)
        except Exception:
            b, a = signal.butter(order, [low / nyq, high / nyq], btype='band')
            filtered = signal.lfilter(b, a, data)

        # Preserve input peak level after filtering
        in_peak = np.max(np.abs(data)) + 1e-12
        out = self._sanitize(filtered, normalize=False)
        out_peak = np.max(np.abs(out)) + 1e-12
        if out_peak > 0:
            out = out * (in_peak / out_peak)
        return out.astype(np.float32)


    def distortion_effect(self, data, drive=1.0):
        out = np.tanh(drive * np.asarray(data, dtype=np.float64))
        return self._sanitize(out, normalize=False)


    def reverb_effect(self, data, decay=0.5, delay_ms=50):
        delay_samples = int(self.sample_rate * delay_ms / 1000)
        if delay_samples <= 0:
            return data

        out = np.copy(data)
        for i in range(delay_samples, len(out)):
            out[i] += decay * out[i - delay_samples]

        # Normalize and sanitize to avoid runaway growth
        out = self._sanitize(out, normalize=True)
        return out


def main():
    synth = Synthesizer(sample_rate=8000)

    t = np.linspace(0, 1, synth.sample_rate, endpoint=False)
    original_waveform = np.sin(2*np.pi*440*t) + 0.5*np.sin(2*np.pi*2200*t)

    original_waveform = synth.lowpass_filter(original_waveform, cutoff_hz=1500.0)

    X = np.fft.rfft(original_waveform)
    N = len(original_waveform)

    output = synth.additive_synth(X, N, duration=3.0)

    sd.play(output, samplerate=synth.sample_rate)
    sd.wait()


if __name__ == "__main__":
    main()
