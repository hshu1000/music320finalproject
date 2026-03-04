from flask import Flask, render_template, request, jsonify
import threading
import os

import freq_processing as fp
from pose_detect import start_pose_detection
from plotter import init_plot

app = Flask(__name__)

# ----------------------------
# Background synth runner state
# ----------------------------
_runner_lock = threading.Lock()
_runner_thread = None
_runner_started = False

def _synth_runner():
    """
    Runs your existing pipeline:
    - matplotlib plot window
    - audio output stream
    - webcam pose detection loop (OpenCV window)
    """
    init_plot()
    fp.start_audio_thread()
    start_pose_detection()

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
# Helpers
# ----------------------------
def current_state():
    # You can expand this later (show more runtime values).
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
    }

# ----------------------------
# Routes
# ----------------------------
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", state=current_state())

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
    # optional time update
    if "time" in payload and payload["time"] is not None:
        fp.set_pedal_time(payload["time"])
    return jsonify({"ok": True, "state": current_state()})

@app.route("/api/flanger", methods=["POST"])
def api_flanger():
    payload = request.json or {}
    on = bool(payload.get("on", False))
    fp.set_flanger_on(on)
    # optional param updates
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
    app.run(debug=True)