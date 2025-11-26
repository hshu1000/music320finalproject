import cv2
from ultralytics import YOLO

from freq_processing import update_audio_from_multiple, pose_to_waveform, _wave_to_period
from plotter import update_plot


def start_pose_detection():
    model = YOLO("yolov8s-pose.pt")
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Fix mirrored webcam feed
        frame = cv2.flip(frame, 1)
        
        results = model(frame, verbose=False)

        waves_this_frame = []

        for r in results:
            if r.keypoints is None:
                continue

            keypoints = r.keypoints.xy

            # Compute per-person overlay spacing based on number of detected people
            num_people = len(keypoints) if keypoints is not None else 1
            # Ensure a reasonable minimum spacing; scale with frame height so overlays fit
            spacing = max(100, frame.shape[0] // max(1, num_people + 1))

            for pi, person in enumerate(keypoints):
                L_shoulder = tuple(person[5].int().tolist())
                R_shoulder = tuple(person[6].int().tolist())
                L_elbow    = tuple(person[7].int().tolist())
                R_elbow    = tuple(person[8].int().tolist())
                L_wrist    = tuple(person[9].int().tolist())
                R_wrist    = tuple(person[10].int().tolist())

                mid = (
                    int((L_shoulder[0] + R_shoulder[0]) / 2),
                    int((L_shoulder[1] + R_shoulder[1]) / 2)
                )
                # draw shoulder-mid
                cv2.circle(frame, mid, 10, (0, 255, 255), -1)

                # helper to interpolate a point between a and b
                def interp(a, b, t=0.5):
                    return (int(a[0] + (b[0] - a[0]) * t), int(a[1] + (b[1] - a[1]) * t))

                # compute extra points along each arm: upper-arm mid (between shoulder-mid and elbow)
                # and forearm mid (between elbow and wrist). This increases control resolution.
                L_upper_mid = interp(mid, L_elbow, 0.5)
                L_forearm_mid = interp(L_elbow, L_wrist, 0.5)
                R_upper_mid = interp(mid, R_elbow, 0.5)
                R_forearm_mid = interp(R_elbow, R_wrist, 0.5)

                # draw skeleton lines and points (including the new interpolated points)
                cv2.line(frame, mid, L_elbow, (255, 0, 0), 3)
                cv2.line(frame, L_elbow, L_wrist, (255, 0, 0), 3)
                cv2.line(frame, mid, R_elbow, (255, 0, 0), 3)
                cv2.line(frame, R_elbow, R_wrist, (255, 0, 0), 3)

                for p in [L_elbow, L_wrist, R_elbow, R_wrist]:
                    cv2.circle(frame, p, 6, (0, 255, 0), -1)

                # show interpolated points in a different color
                for p in [L_upper_mid, L_forearm_mid, R_upper_mid, R_forearm_mid]:
                    cv2.circle(frame, p, 5, (0, 128, 255), -1)

                # build the point list including extra points. Convert to floats for downstream processing.
                pts = [
                    L_wrist,
                    L_forearm_mid,
                    L_elbow,
                    L_upper_mid,
                    mid,
                    R_upper_mid,
                    R_elbow,
                    R_forearm_mid,
                    R_wrist
                ]
                pts = [tuple(map(float, p)) for p in pts]

                # Sort points by x-coordinate (left to right) to ensure lines don't overlap when tiled
                pts_sorted = sorted(pts, key=lambda p: p[0])

                # Draw connecting lines in sorted x-order
                for i in range(len(pts_sorted) - 1):
                    p1 = (int(pts_sorted[i][0]), int(pts_sorted[i][1]))
                    p2 = (int(pts_sorted[i+1][0]), int(pts_sorted[i+1][1]))
                    cv2.line(frame, p1, p2, (200, 200, 0), 2)

                # Compute metadata: average y-coordinate of all points
                avg_y = sum(p[1] for p in pts_sorted) / len(pts_sorted)

                # include actual frame height so downstream mapping isn't guessed
                frame_h = frame.shape[0]

                # Append metadata tuple to the end of pts_sorted: (avg_y, frame_height)
                metadata = (avg_y, frame_h)
                pts_with_metadata = list(pts_sorted) + [metadata]

                # pose_to_waveform now returns (original_wave, freq, filtered_wave, cutoff, norm)
                wave, freq, filtered_wave, cutoff, norm = pose_to_waveform(pts_with_metadata)

                # For audio, use the filtered waveform; for plotting keep both
                waves_this_frame.append((filtered_wave, freq, wave))

                # Draw an on-screen overlay showing the cutoff and normalized control value
                # Offset vertically by person index so multiple people don't overlap
                try:
                    overlay_x = 10
                    overlay_y = 30 + pi * spacing
                    font = cv2.FONT_HERSHEY_SIMPLEX

                    # Prepare lines and font parameters
                    title = f"Person {pi+1}"
                    line1 = f"Cutoff: {cutoff:.0f} Hz"
                    line2 = f"Norm: {norm:.2f}"
                    line3 = f"AvgY: {avg_y:.1f}"
                    lines = [title, line1, line2, line3]
                    scales = [0.7, 0.7, 0.6, 0.6]
                    thicks = [2, 2, 1, 1]

                    # Measure text sizes to compute rectangle size
                    widths = []
                    heights = []
                    baselines = []
                    for txt, sc, th in zip(lines, scales, thicks):
                        (w, h), base = cv2.getTextSize(txt, font, sc, th)
                        widths.append(w)
                        heights.append(h)
                        baselines.append(base)

                    max_w = max(widths)
                    total_h = sum(heights) + (len(lines) - 1) * 6

                    # Rectangle coordinates (with padding)
                    pad_x = 10
                    pad_y = 8
                    rx1 = max(0, overlay_x - pad_x)
                    ry1 = max(0, overlay_y - pad_y - heights[0])
                    rw = max_w + pad_x * 2
                    rh = total_h + pad_y * 2
                    rx2 = min(frame.shape[1], rx1 + rw)
                    ry2 = min(frame.shape[0], ry1 + rh)

                    # Draw semi-transparent rectangle by blending an overlay
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (rx1, ry1), (rx2, ry2), (0, 0, 0), -1)
                    alpha = 0.45
                    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

                    # Draw each text line over the blended rectangle
                    y = ry1 + pad_y + heights[0]
                    for i, txt in enumerate(lines):
                        sc = scales[i]
                        th = thicks[i]
                        color = (50, 220, 255) if i == 0 else (0, 200, 255)
                        cv2.putText(frame, txt, (overlay_x, y), font, sc, color, th, cv2.LINE_AA)
                        y += heights[i] + 6
                except Exception:
                    pass

        if waves_this_frame:
            # waves_this_frame entries: (filtered_wave, freq, original_wave)
            audio_list = [(fw, f) for (fw, f, ow) in waves_this_frame]
            update_audio_from_multiple(audio_list)

            processed = []
            for (fw, f, ow) in waves_this_frame:
                orig_period = _wave_to_period(ow, f)
                filt_period = _wave_to_period(fw, f)
                # pass tuple (original, filtered) so plotter can draw both
                processed.append((orig_period, filt_period))

            update_plot(processed)


        cv2.imshow("Multi-Person Arm Node Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()