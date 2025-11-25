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

            for person in keypoints:
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

                # Append metadata tuple to the end of pts_sorted
                metadata = (avg_y,)
                pts_with_metadata = list(pts_sorted) + [metadata]

                wave, freq = pose_to_waveform(pts_with_metadata)
                waves_this_frame.append((wave, freq))

        if waves_this_frame:
            update_audio_from_multiple(waves_this_frame)
            processed = []
            for wave, freq in waves_this_frame:
                period = _wave_to_period(wave, freq)
                processed.append(period)

            update_plot(processed)


        cv2.imshow("Multi-Person Arm Node Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()