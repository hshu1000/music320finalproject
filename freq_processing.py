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

        waveform /= np.max(np.abs(waveform) + 1e-12)
        return waveform.astype(np.float32)


    def additive_synth(self, X, original_length, duration):
        freqs = np.fft.rfftfreq(original_length, d=1 / self.sample_rate)

        amps = (2.0 / original_length) * np.abs(X)
        amps[0] /= 2.0 
        if original_length % 2 == 0:
            amps[-1] /= 2.0

        phases = np.angle(X)

        return self.additive_synth_helper(freqs, amps, phases, duration)


    def lowpass_filter(self, data, cutoff_hz, order=4):
        nyq = 0.5 * self.sample_rate
        b, a = signal.butter(order, cutoff_hz / nyq, btype='low')
        return signal.lfilter(b, a, data).astype(np.float32)


    def highpass_filter(self, data, cutoff_hz, order=4):
        nyq = 0.5 * self.sample_rate
        b, a = signal.butter(order, cutoff_hz / nyq, btype='high')
        return signal.lfilter(b, a, data).astype(np.float32)


    def bandpass_filter(self, data, low, high, order=4):
        nyq = 0.5 * self.sample_rate
        b, a = signal.butter(order, [low / nyq, high / nyq], btype='band')
        return signal.lfilter(b, a, data).astype(np.float32)


    def distortion_effect(self, data, drive=1.0):
        return np.tanh(drive * data).astype(np.float32)


    def reverb_effect(self, data, decay=0.5, delay_ms=50):
        delay_samples = int(self.sample_rate * delay_ms / 1000)
        if delay_samples <= 0:
            return data

        out = np.copy(data)
        for i in range(delay_samples, len(out)):
            out[i] += decay * out[i - delay_samples]

        out /= np.max(np.abs(out) + 1e-12)
        return out.astype(np.float32)


def main():
    synth = Synthesizer(sample_rate=8000)

    t = np.linspace(0, 1, synth.sample_rate, endpoint=False)
    original_waveform = np.sin(2*np.pi*440*t) + 0.5*np.sin(2*np.pi*2200*t)

    X = np.fft.rfft(original_waveform)
    N = len(original_waveform)

    output = synth.additive_synth(X, N, duration=3.0)

    sd.play(output, samplerate=synth.sample_rate)
    sd.wait()


if __name__ == "__main__":
    main()
