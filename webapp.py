import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
from pose_detect import detect_arms  # your existing detection function

# ---- Page setup ----
st.set_page_config(page_title="Pose Detection", page_icon="🎥", layout="wide")

# ---- Custom CSS ----
st.markdown(
    """
    <style>
    /* Base page layout */
    .stApp {
        font-family: "Inter", sans-serif;
        transition: background-color 0.4s ease, color 0.4s ease;
    }

    /* Theme auto-detection */
    @media (prefers-color-scheme: dark) {
        .stApp {
            background-color: #0E1117;
            color: white;
        }
        .title {
            color: #1DB954;
        }
        .css-1v0mbdj button {
            background-color: #1DB954 !important;
            color: white !important;
        }
    }

    @media (prefers-color-scheme: light) {
        .stApp {
            background-color: #F9FAFB;
            color: #111827;
        }
        .title {
            color: #1A202C;
        }
        .css-1v0mbdj button {
            background-color: #4F46E5 !important;
            color: white !important;
        }
    }

    /* Centered title */
    .title {
        text-align: center;
        font-size: 2.5rem;
        font-weight: 800;
        margin-top: 0.5em;
        margin-bottom: 0.5em;
    }

    /* Hide default video controls (time bar, pause, etc.) */
    video {
        -webkit-media-controls-enclosure {
            display: none !important;
        }
        outline: none;
        border-radius: 12px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.3);
        margin: auto;
        display: block;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(0,0,0,0.04);
        backdrop-filter: blur(10px);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---- Title ----
st.markdown('<h1 class="title">🎥 Multi-Person Arm Node Detection WebApp</h1>', unsafe_allow_html=True)

# ---- Sidebar Controls ----
st.sidebar.header("⚙️ Settings")
show_coords = st.sidebar.checkbox("Show Sorted Coordinates in Console", value=True)


# ---- Streamlit WebRTC Integration ----
class PoseTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        annotated, person_coords = detect_arms(img)

        if show_coords:
            for i, coords in enumerate(person_coords, start=1):
                print(f"Person {i}:")
                print(f"  x_sorted: {coords['x_sorted']}")
                print(f"  y_sorted: {coords['y_sorted']}\n")

        return annotated


# ---- WebRTC Stream ----
webrtc_streamer(
    key="pose",
    video_transformer_factory=PoseTransformer,
    media_stream_constraints={"video": True, "audio": False},
)
