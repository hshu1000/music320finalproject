import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

from ultralytics import YOLO

# These come from your existing pipeline
from freq_processing import (
    start_audio_thread,
    update_audio_from_multiple,
    _wave_to_period,
)

# Optional, in case you still want the desktop matplotlib plot window
# from plotter import init_plot, update_plot


# If your MIN_FREQ lives in freq_processing you can import it instead
MIN_FREQ = 261.62  # middle C


# Simple helper to map an arm angle to a frequency and one period of a sine wave
def angle_to_wave(angle_rad: float, sample_rate: int = 44100):
    """
    Map arm angle to frequency and return a single period waveform.

    angle_rad is in [-pi, pi].
    We map that range to [MIN_FREQ, MIN_FREQ + 600] for fun.
    """
    # normalize angle to [0, 1]
    norm = (angle_rad + np.pi) / (2.0 * np.pi)
    freq = MIN_FREQ + norm * 600.0

    # build one period of a sine wave
    period_len = int(sample_rate / freq)
    if period_len < 16:
        period_len = 16

    t = np.linspace(0.0, 1.0 / freq, period_len, endpoint=False)
    wave = np.sin(2.0 * np.pi * freq * t).astype(np.float32)

    return wave, freq


def compute_arm_angle(keypoints_person, left=True):
    """
    Compute arm angle from YOLOv8 pose keypoints.

    keypoints_person is shape (num_keypoints, 3).
    We use (shoulder, wrist) for one arm.

    YOLOv8 pose keypoint index reference for COCO:
      5  = left shoulder
      7  = left elbow
      9  = left wrist
      6  = right shoulder
      8  = right elbow
      10 = right wrist
    """
    if left:
        shoulder_idx = 5
        wrist_idx = 9
    else:
        shoulder_idx = 6
        wrist_idx = 10

    shoulder = keypoints_person[shoulder_idx, :2]
    wrist = keypoints_person[wrist_idx, :2]

    # if any of them are missing (0,0) or low confidence, bail
    if np.allclose(shoulder, 0) or np.allclose(wrist, 0):
        return None

    vec = wrist - shoulder  # in image coordinates
    # y axis is down in images, so flip sign to make angles more intuitive
    angle = np.arctan2(-vec[1], vec[0])
    return angle


# Basic STUN server config so WebRTC can connect
RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
        ]
    }
)


class PoseSynthTransformer:
    """
    Video transformer for streamlit-webrtc.

    For each frame:
      - run YOLOv8 pose
      - draw skeleton
      - compute one waveform per person based on arm angle
      - call your freq_processing.update_audio_from_multiple
    """

    def __init__(self):
        # Load model once
        self.model = YOLO("yolov8s-pose.pt")
        # Start your audio thread once
        start_audio_thread()
        # Optional: start matplotlib plot window
        # init_plot()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        # Run pose model
        results = self.model(img, verbose=False)
        waves_this_frame = []

        for result in results:
            kpts = result.keypoints
            if kpts is None:
                continue

            # YOLOv8 keypoints tensor: (num_people, num_kpts, 3)
            kpts_np = kpts.data.cpu().numpy()

            for person in kpts_np:
                # Draw keypoints and skeleton for visualization
                for (x, y, conf) in person:
                    if conf < 0.3:
                        continue
                    cv2.circle(img, (int(x), int(y)), 3, (0, 255, 0), -1)

                # Compute arm angle (left arm for now)
                angle = compute_arm_angle(person, left=True)
                if angle is None:
                    continue

                # Map angle to wave and frequency
                wave, freq = angle_to_wave(angle)

                # Add to list for this frame
                waves_this_frame.append((wave, freq))

                # Visual feedback: draw text with frequency
                cv2.putText(
                    img,
                    f"{freq:.1f} Hz",
                    (int(person[5, 0]), int(person[5, 1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        # Send waves to your audio engine if any
        if len(waves_this_frame) > 0:
            update_audio_from_multiple(waves_this_frame)

            # If you want to keep using the existing matplotlib window
            # you can compute periods and call update_plot here
            # processed = []
            # for wave, freq in waves_this_frame:
            #     period = _wave_to_period(wave, freq)
            #     processed.append(period)
            # update_plot(processed)

        # Return annotated video frame
        return av.VideoFrame.from_ndarray(img, format="bgr24")


def main():
    st.set_page_config(
        page_title="Pose Synth Webapp",
        layout="wide",
    )

    st.title("Pose Synth Webapp")
    st.write(
        "Raise and lower your arm, the angle controls the frequency of a synthesized tone."
    )
    st.write(
        "This app uses YOLOv8 pose for keypoints, your existing audio thread in freq_processing, "
        "and streamlit-webrtc for live video."
    )

    webrtc_streamer(
        key="pose-synth",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=PoseSynthTransformer,
    )

    st.markdown(
        """
        **Notes**

        - Audio output is handled by your existing `sounddevice` audio callback in `freq_processing.start_audio_thread`.
        - This webapp only handles video in the browser. The sound still plays through your local machine, same as your original script.
        - If you want inline waveform or FFT plots in the Streamlit page, we can store the latest waveform in a global
          and display it with `st.pyplot` in a side column.
        """
    )


if __name__ == "__main__":
    main()
