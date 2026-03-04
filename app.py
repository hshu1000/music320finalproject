from flask import Flask, render_template, request, jsonify, Response
import threading
import os
import time

import cv2
import mediapipe as mp
from ultralytics import YOLO

import freq_processing as fp

app = Flask(__name__)

# ----------------------------
# Background synth runner state
# ----------------------------
_runner_lock = threading.Lock()
_runner_thread = None
_runner_started = False

def _synth_runner():
    """
    Runs your existing pipeline (safe subset on macOS):
    - audio output stream
    Camera + pose processing happens inside the MJPEG stream generator.
    """
    fp.start_audio_thread()

def start_synth_once():
    global _runner_thread, _runner_started
    with _runner_lock:
        if _runner_started and _runner_thread and _runner_thread.is_alive():
            return False
        _runner_started = True
        _runner_thread = threading.Thread(target=_synth_runner, daemon=True)
        _runner_thread.start()
        return True


# ----------------------------
# Pose models (init once)
# ----------------------------
_mp_hands = mp.solutions.hands
_mp_draw = mp.solutions.drawing_utils

_hands = _mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=10,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

_yolo_pose_model = None
_yolo_lock = threading.Lock()

def _get_yolo_pose_model():
    global _yolo_pose_model
    with _yolo_lock:
        if _yolo_pose_model is None:
            _yolo_pose_model = YOLO("yolov8s-pose.pt")
    return _yolo_pose_model


# ----------------------------
# Camera stream (MJPEG)
# ----------------------------
_cam_lock = threading.Lock()
_cam = None
_cam_on = False

def _open_camera():
    global _cam
    if _cam is None:
        _cam = cv2.VideoCapture(0)
        # Optional resolution
        # _cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        # _cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return _cam

def _close_camera():
    global _cam
    if _cam is not None:
        try:
            _cam.release()
        except Exception:
            pass
        _cam = None


def _interp(a, b, t=0.5):
    return (int(a[0] + (b[0] - a[0]) * t), int(a[1] + (b[1] - a[1]) * t))


def gen_frames():
    """
    Generator that yields MJPEG frames for <img src="/video_feed">.
    Also runs your original pose->audio logic every frame:
      - hand mode: MediaPipe
      - arm mode: YOLOv8 pose
      - drives fp.update_audio_from_multiple(...)
    """
    global _cam_on

    TIP_IDXS = [4, 8, 12, 16, 20]  # thumb + fingertips

    while True:
        with _cam_lock:
            if not _cam_on:
                break
            cap = _open_camera()

        ok, frame = cap.read()
        if not ok or frame is None or frame.size == 0:
            time.sleep(0.01)
            continue

        frame = cv2.flip(frame, 1)

        waves_this_frame = []

        # Mode banner (optional)
        mode_txt = "HAND MODE" if getattr(fp, "CURRENT_MODE", "hand") == "hand" else "ARM MODE"
        cv2.putText(
            frame, mode_txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            1.0, (0, 255, 255), 2, cv2.LINE_AA
        )

        # ----------------------------
        # HAND MODE (MediaPipe)
        # ----------------------------
        if getattr(fp, "CURRENT_MODE", "hand") == "hand":
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = _hands.process(rgb)

            if result.multi_hand_landmarks:
                for i, handLm in enumerate(result.multi_hand_landmarks):
                    pts = []
                    for idx in TIP_IDXS:
                        lm = handLm.landmark[idx]
                        x = int(lm.x * frame.shape[1])
                        y = int(lm.y * frame.shape[0])
                        pts.append((float(x), float(y)))
                        cv2.circle(frame, (x, y), 6, (0, 200, 255), -1)

                    _mp_draw.draw_landmarks(frame, handLm, _mp_hands.HAND_CONNECTIONS)

                    avg_x = sum(p[0] for p in pts) / len(pts)
                    avg_y = sum(p[1] for p in pts) / len(pts)
                    cv2.circle(frame, (int(avg_x), int(avg_y)), 8, (0, 0, 255), -1)

                    metadata = (avg_y, frame.shape[0], avg_x, frame.shape[1])
                    pts_with_metadata = pts + [metadata]

                    try:
                        wave, freq, note_name, lp_cutoff, lp_bin_idx, hp_cutoff, hp_bin_idx = fp.pose_to_waveform(pts_with_metadata)
                        waves_this_frame.append((wave, freq, lp_cutoff, hp_cutoff, note_name))
                    except Exception:
                        pass

        # ----------------------------
        # ARM MODE (YOLOv8 pose)
        # ----------------------------
        else:
            try:
                model_pose = _get_yolo_pose_model()
                results = model_pose(frame, verbose=False)

                for r in results:
                    if r.keypoints is None:
                        continue

                    pts_per_person = r.keypoints.xy
                    for pi, person in enumerate(pts_per_person):
                        # shoulder, elbow, wrist indices (COCO style in your original file)
                        Ls = tuple(person[5].int().tolist())
                        Rs = tuple(person[6].int().tolist())
                        Le = tuple(person[7].int().tolist())
                        Re = tuple(person[8].int().tolist())
                        Lw = tuple(person[9].int().tolist())
                        Rw = tuple(person[10].int().tolist())

                        mid = (int((Ls[0] + Rs[0]) / 2), int((Ls[1] + Rs[1]) / 2))
                        cv2.circle(frame, mid, 8, (0, 255, 255), -1)

                        L_um = _interp(mid, Le, 0.5)
                        L_fm = _interp(Le, Lw, 0.5)
                        R_um = _interp(mid, Re, 0.5)
                        R_fm = _interp(Re, Rw, 0.5)

                        basic_pts = [Lw, L_fm, Le, L_um, mid, R_um, Re, R_fm, Rw]
                        pts = sorted([tuple(map(float, p)) for p in basic_pts], key=lambda p: p[0])

                        for (x, y) in basic_pts:
                            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

                        avg_x = sum(p[0] for p in pts) / len(pts)
                        avg_y = sum(p[1] for p in pts) / len(pts)
                        cv2.circle(frame, (int(avg_x), int(avg_y)), 8, (0, 0, 255), -1)

                        metadata = (avg_y, frame.shape[0], avg_x, frame.shape[1])
                        pts_with_metadata = pts + [metadata]

                        try:
                            wave, freq, note_name, lp_cutoff, lp_bin_idx, hp_cutoff, hp_bin_idx = fp.pose_to_waveform(pts_with_metadata)
                            waves_this_frame.append((wave, freq, lp_cutoff, hp_cutoff, note_name))
                        except Exception:
                            pass
            except Exception:
                # If YOLO is not available or errors, just do no voices this frame
                pass

        # ----------------------------
        # Drive audio (this is the important part)
        # ----------------------------
        if waves_this_frame:
            audio_list = [(w, f, lp, hp) for (w, f, lp, hp, note) in waves_this_frame]
            fp.update_audio_from_multiple(audio_list)

            # Optional overlay for first voice so you can see sound mapping live
            try:
                w, f, lp, hp, note = waves_this_frame[0]
                cv2.putText(
                    frame,
                    f"{note}  {f:.0f} Hz",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (50, 220, 255),
                    2,
                    cv2.LINE_AA
                )
            except Exception:
                pass
        else:
            fp.update_audio_from_multiple([])

        ok2, buffer = cv2.imencode(".jpg", frame)
        if not ok2:
            continue

        frame_bytes = buffer.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )


# ----------------------------
# Helpers
# ----------------------------
def current_state():
    instruments = getattr(fp, "PERSON_INSTRUMENTS", {})
    return {
        "mode": getattr(fp, "CURRENT_MODE", "hand"),
        "scale": getattr(fp, "CURRENT_SCALE", "c major"),
        "pedal": getattr(fp, "PEDAL_MODE", False),
        "pedal_time": getattr(fp, "PEDAL_TIME", 10.0),
        "flanger": getattr(fp, "FLANGER_ON", False),
        "flanger_rate": getattr(fp, "FLANGER_RATE", 0.2),
        "flanger_depth_ms": getattr(fp, "FLANGER_DEPTH_MS", 2.0),
        "instruments": instruments,
        "available_instruments": fp.list_instruments() if hasattr(fp, "list_instruments") else [],
        "recording": getattr(fp, "RECORDING", False),
        "camera_on": _cam_on,
    }


# ----------------------------
# Routes
# ----------------------------
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", state=current_state())

@app.route("/video_feed")
def video_feed():
    global _cam_on
    with _cam_lock:
        _cam_on = True
        _open_camera()
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/camera/start", methods=["POST"])
def api_camera_start():
    global _cam_on
    with _cam_lock:
        _cam_on = True
        _open_camera()
    return jsonify({"ok": True, "state": current_state()})

@app.route("/api/camera/stop", methods=["POST"])
def api_camera_stop():
    global _cam_on
    with _cam_lock:
        _cam_on = False
        _close_camera()
    return jsonify({"ok": True, "state": current_state()})

@app.route("/api/state", methods=["GET"])
def api_state():
    return jsonify(current_state())

@app.route("/api/start", methods=["POST"])
def api_start():
    started = start_synth_once()
    return jsonify({"ok": True, "started": started, "state": current_state()})

@app.route("/api/mode", methods=["POST"])
def api_mode():
    mode = (request.json or {}).get("mode", "").strip().lower()
    fp.set_global_mode(mode)
    return jsonify({"ok": True, "state": current_state()})

@app.route("/api/scale", methods=["POST"])
def api_scale():
    scale = (request.json or {}).get("scale", "").strip()
    fp.set_global_scale(scale)
    return jsonify({"ok": True, "state": current_state()})

@app.route("/api/instrument", methods=["POST"])
def api_instrument():
    payload = request.json or {}
    person = int(payload.get("person", 1))
    name = str(payload.get("name", "")).strip().lower()
    ok = fp.set_person_instrument(person, name)
    return jsonify({"ok": bool(ok), "state": current_state()})

@app.route("/api/pedal", methods=["POST"])
def api_pedal():
    payload = request.json or {}
    on = bool(payload.get("on", False))
    fp.set_pedal_mode(on)
    if "time" in payload and payload["time"] is not None:
        fp.set_pedal_time(payload["time"])
    return jsonify({"ok": True, "state": current_state()})

@app.route("/api/flanger", methods=["POST"])
def api_flanger():
    payload = request.json or {}
    on = bool(payload.get("on", False))
    fp.set_flanger_on(on)
    rate = payload.get("rate", None)
    depth = payload.get("depth_ms", None)
    if rate is not None or depth is not None:
        fp.set_flanger_params(rate=rate, depth_ms=depth)
    return jsonify({"ok": True, "state": current_state()})

@app.route("/api/record/start", methods=["POST"])
def api_record_start():
    payload = request.json or {}
    filename = str(payload.get("filename", "take.wav")).strip()
    fp.start_recording(filename)
    return jsonify({"ok": True, "state": current_state()})

@app.route("/api/record/stop", methods=["POST"])
def api_record_stop():
    fp.stop_recording()
    return jsonify({"ok": True, "state": current_state()})

@app.route("/api/record/list", methods=["GET"])
def api_record_list():
    os.makedirs("recordings", exist_ok=True)
    files = sorted([f for f in os.listdir("recordings") if f.lower().endswith(".wav")])
    return jsonify({"ok": True, "files": files})

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)