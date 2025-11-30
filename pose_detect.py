import cv2
from ultralytics import YOLO
import mediapipe as mp

import freq_processing as fp
from plotter import update_plot


def start_pose_detection():
    model_pose = YOLO('yolov8s-pose.pt')

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=10,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)

    # Main loop
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        waves_this_frame = []

        # Use CURRENT_MODE from freq_processing to decide hand vs arm
        mode_txt = 'HAND MODE' if fp.CURRENT_MODE == 'hand' else 'ARM MODE'
        cv2.putText(frame, mode_txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 255), 2, cv2.LINE_AA)

        # Hand mode
        if fp.CURRENT_MODE == 'hand':
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

                    # average position
                    avg_x = sum(p[0] for p in pts) / len(pts)
                    avg_y = sum(p[1] for p in pts) / len(pts)

                    # draw avg point on camera feed
                    cv2.circle(frame,
                               (int(avg_x), int(avg_y)),
                               10, (0, 0, 255), -1)

                    # add metadata for vertical & horizontal control
                    metadata = (avg_y, frame.shape[0], avg_x, frame.shape[1])
                    pts_with_metadata = pts + [metadata]

                    (wave,
                     freq,
                     note_name,
                     lp_cutoff,
                     lp_bin_idx,
                     hp_cutoff,
                     hp_bin_idx) = fp.pose_to_waveform(pts_with_metadata)

                    waves_this_frame.append(
                        (wave, freq, lp_cutoff, hp_cutoff, note_name, lp_bin_idx, hp_bin_idx)
                    )

                    overlay_y = 60 + i * spacing
                    overlay_lines(frame,
                                  i,
                                  freq,
                                  note_name,
                                  lp_cutoff,
                                  lp_bin_idx,
                                  hp_cutoff,
                                  hp_bin_idx,
                                  overlay_y)

        # Arm mode
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

                    # average position
                    avg_x = sum(p[0] for p in pts) / len(pts)
                    avg_y = sum(p[1] for p in pts) / len(pts)

                    # draw avg point on camera feed
                    cv2.circle(frame,
                               (int(avg_x), int(avg_y)),
                               10, (0, 0, 255), -1)

                    # metadata
                    metadata = (avg_y, frame.shape[0], avg_x, frame.shape[1])
                    pts_with_metadata = pts + [metadata]

                    (wave,
                     freq,
                     note_name,
                     lp_cutoff,
                     lp_bin_idx,
                     hp_cutoff,
                     hp_bin_idx) = fp.pose_to_waveform(pts_with_metadata)

                    waves_this_frame.append(
                        (wave, freq, lp_cutoff, hp_cutoff, note_name, lp_bin_idx, hp_bin_idx)
                    )

                    overlay_y = 60 + pi * spacing
                    overlay_lines(frame,
                                  pi,
                                  freq,
                                  note_name,
                                  lp_cutoff,
                                  lp_bin_idx,
                                  hp_cutoff,
                                  hp_bin_idx,
                                  overlay_y)

        # Audio and plotting
        if waves_this_frame:
            # audio: each entry is (wave, freq, lp_cutoff, hp_cutoff)
            audio_list = [(w, f, lp, hp) for (w, f, lp, hp, note, lb, hb) in waves_this_frame]
            fp.update_audio_from_multiple(audio_list)

            # plotting: show the actual instrument-shaped period
            processed = []
            for idx, (w, f, lp, hp, note, lb, hb) in enumerate(waves_this_frame):
                base = fp._wave_to_period(w, f)
                instr_name = fp.get_instrument_for_person(idx + 1)
                shaped = fp.apply_instrument_profile(base, instr_name)
                processed.append(shaped)

            update_plot(processed)

        else:
            # <<< THIS IS THE IMPORTANT PART >>>
            # No hands / arms detected this frame → hard stop all audio
            fp.update_audio_from_multiple([])

        cv2.imshow('Hand/Arm Pose Detection', frame)

        key = cv2.waitKey(1) & 0xFF

        # ESC to quit
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


def overlay_lines(frame,
                  pi,
                  freq,
                  note_name,
                  lp_cutoff,
                  lp_bin_idx,
                  hp_cutoff,
                  hp_bin_idx,
                  overlay_y):
    font = cv2.FONT_HERSHEY_SIMPLEX

    def bin_bar(bin_idx, num_bins=8):
        return '[' + ''.join('#' if i <= bin_idx else '-' for i in range(num_bins)) + ']'

    lp_bar = bin_bar(lp_bin_idx)
    hp_bar = bin_bar(hp_bin_idx)

    # Instrument label for this person index (1-based)
    person_index = pi + 1
    instr = fp.get_instrument_for_person(person_index)

    lines = [
        f'Person {person_index} ({instr})',
        f'Freq: {freq:.0f} Hz ({note_name})',
        f'LP: {lp_cutoff:.0f} Hz {lp_bar}',
        f'HP: {hp_cutoff:.0f} Hz {hp_bar}',
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
