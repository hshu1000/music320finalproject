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

                cv2.circle(frame, mid, 10, (0, 255, 255), -1)
                cv2.line(frame, mid, L_elbow, (255, 0, 0), 3)
                cv2.line(frame, L_elbow, L_wrist, (255, 0, 0), 3)
                cv2.line(frame, mid, R_elbow, (255, 0, 0), 3)
                cv2.line(frame, R_elbow, R_wrist, (255, 0, 0), 3)

                for p in [L_elbow, L_wrist, R_elbow, R_wrist]:
                    cv2.circle(frame, p, 6, (0, 255, 0), -1)

                pts = [
                    L_wrist,
                    L_elbow,
                    mid,
                    R_elbow,
                    R_wrist
                ]
                pts = [tuple(map(float, p)) for p in pts]

                wave, freq = pose_to_waveform(pts)
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
