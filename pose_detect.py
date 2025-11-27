import cv2
from ultralytics import YOLO
import mediapipe as mp

from freq_processing import update_audio_from_multiple, pose_to_waveform, _wave_to_period
from plotter import update_plot


def start_pose_detection():
    # ------------------------------
    # INITIAL SETUP
    # ------------------------------
    model_pose = YOLO("yolov8s-pose.pt")

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=10,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)

    # MODE TOGGLE
    hand_mode = True  # False = arm mode, True = hand mode
    # If keypress is unreliable, just manually set:
    # hand_mode = True   # force hand mode
    # hand_mode = False  # force arm mode

    # ------------------------------
    # MAIN LOOP
    # ------------------------------
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        waves_this_frame = []

        # --------------------------------------------------------
        # MODE SWITCH
        # --------------------------------------------------------
        mode_txt = "HAND MODE" if hand_mode else "ARM MODE"
        cv2.putText(frame, mode_txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 255), 2, cv2.LINE_AA)

        # --------------------------------------------------------
        # HAND MODE
        # --------------------------------------------------------
        if hand_mode:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                spacing = frame.shape[0] // (len(result.multi_hand_landmarks) + 1)

                for i, handLm in enumerate(result.multi_hand_landmarks):
                    # fingertip indices
                    TIP_IDXS = [4, 8, 12, 16, 20]  # thumb & fingertips

                    pts = []
                    for idx in TIP_IDXS:
                        lm = handLm.landmark[idx]
                        x = int(lm.x * frame.shape[1])
                        y = int(lm.y * frame.shape[0])
                        pts.append((float(x), float(y)))
                        cv2.circle(frame, (x, y), 8, (0, 200, 255), -1)

                    # draw fingertip connections (simple chain)
                    for a, b in zip(pts[:-1], pts[1:]):
                        cv2.line(frame,
                                 (int(a[0]), int(a[1])),
                                 (int(b[0]), int(b[1])),
                                 (255, 150, 0), 2)

                    # add metadata for vertical control (similar to arm mode)
                    avg_y = sum(p[1] for p in pts) / len(pts)
                    metadata = (avg_y, frame.shape[0])
                    pts_with_metadata = pts + [metadata]

                    wave, freq, filtered_wave, cutoff, norm = pose_to_waveform(pts_with_metadata)
                    waves_this_frame.append((filtered_wave, freq, wave))

                    # overlay like in arm mode
                    overlay_y = 60 + i * spacing
                    overlay_lines(frame, i, cutoff, norm, avg_y, overlay_y)

        # --------------------------------------------------------
        # ARM MODE
        # --------------------------------------------------------
        else:
            results = model_pose(frame, verbose=False)

            for r in results:
                if r.keypoints is None:
                    continue

                pts_per_person = r.keypoints.xy
                num_people = len(pts_per_person)
                spacing = max(100, frame.shape[0] // (num_people + 1))

                for pi, person in enumerate(pts_per_person):

                    # shoulder, elbow, wrist
                    Ls = tuple(person[5].int().tolist())
                    Rs = tuple(person[6].int().tolist())
                    Le = tuple(person[7].int().tolist())
                    Re = tuple(person[8].int().tolist())
                    Lw = tuple(person[9].int().tolist())
                    Rw = tuple(person[10].int().tolist())

                    arm_joints = [Le, Re, Lw, Rw]
                    for (x, y) in arm_joints:
                        cv2.circle(frame, (x, y), 8, (0, 255, 0), -1)

                    mid = (int((Ls[0] + Rs[0]) / 2), int((Ls[1] + Rs[1]) / 2))
                    cv2.circle(frame, mid, 10, (0, 255, 255), -1)

                    def interp(a, b, t=0.5):
                        return (int(a[0] + (b[0] - a[0]) * t),
                                int(a[1] + (b[1] - a[1]) * t))

                    L_um = interp(mid, Le, 0.5)
                    L_fm = interp(Le, Lw, 0.5)
                    R_um = interp(mid, Re, 0.5)
                    R_fm = interp(Re, Rw, 0.5)

                    basic_pts = [
                        Lw, L_fm, Le, L_um, mid,
                        R_um, Re, R_fm, Rw
                    ]
                    pts = sorted([tuple(map(float, p)) for p in basic_pts], key=lambda p: p[0])

                    # draw connections
                    for a, b in zip(pts[:-1], pts[1:]):
                        cv2.line(frame,
                                 (int(a[0]), int(a[1])),
                                 (int(b[0]), int(b[1])),
                                 (200, 200, 0), 2)

                    # metadata
                    avg_y = sum(p[1] for p in pts) / len(pts)
                    pts_with_metadata = pts + [(avg_y, frame.shape[0])]

                    wave, freq, filtered_wave, cutoff, norm = pose_to_waveform(pts_with_metadata)
                    waves_this_frame.append((filtered_wave, freq, wave))

                    overlay_y = 60 + pi * spacing
                    overlay_lines(frame, pi, cutoff, norm, avg_y, overlay_y)

        # --------------------------------------------------------
        # AUDIO + PLOTTING
        # --------------------------------------------------------
        if waves_this_frame:
            audio_list = [(fw, f) for (fw, f, ow) in waves_this_frame]
            update_audio_from_multiple(audio_list)

            processed = []
            for (fw, f, ow) in waves_this_frame:
                orig = _wave_to_period(ow, f)
                filt = _wave_to_period(fw, f)
                processed.append((orig, filt))

            update_plot(processed)

        # --------------------------------------------------------
        # DISPLAY + KEYPRESS
        # --------------------------------------------------------
        cv2.imshow("Hand/Arm Pose Detection", frame)

        # NOTE: must be AFTER imshow and inside OpenCV window focus
        key = cv2.waitKey(1) & 0xFF
        if key == ord('h'):
            hand_mode = not hand_mode
            print("Toggled mode:", "HAND MODE" if hand_mode else "ARM MODE")

        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


# -----------------------------------------------------------
# SMALL HELPER FUNCTION
# -----------------------------------------------------------
def overlay_lines(frame, pi, cutoff, norm, avg_y, overlay_y):
    font = cv2.FONT_HERSHEY_SIMPLEX
    lines = [
        f"Person {pi+1}",
        f"Cutoff: {cutoff:.0f} Hz",
        f"Norm: {norm:.2f}",
        f"AvgY: {avg_y:.1f}",
    ]
    scales = [0.7, 0.7, 0.6, 0.6]
    thicks = [2, 2, 1, 1]

    widths = []
    heights = []
    bases = []
    for txt, sc, th in zip(lines, scales, thicks):
        (w, h), base = cv2.getTextSize(txt, font, sc, th)
        widths.append(w)
        heights.append(h)
        bases.append(base)

    max_w = max(widths)
    total_h = sum(heights) + (len(lines) - 1) * 6

    pad_x = 10
    pad_y = 8

    rx1 = 10
    ry1 = max(0, overlay_y - pad_y - heights[0])
    rw = max_w + pad_x * 2
    rh = total_h + pad_y * 2
    rx2 = min(frame.shape[1], rx1 + rw)
    ry2 = min(frame.shape[0], ry1 + rh)

    overlay = frame.copy()
    cv2.rectangle(overlay, (rx1, ry1), (rx2, ry2), (0, 0, 0), -1)
    alpha = 0.45
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    y = ry1 + pad_y + heights[0]
    for i, txt in enumerate(lines):
        color = (50, 220, 255) if i == 0 else (0, 200, 255)
        cv2.putText(frame, txt, (rx1 + pad_x, y), font,
                    scales[i], color, thicks[i], cv2.LINE_AA)
        y += heights[i] + 6
