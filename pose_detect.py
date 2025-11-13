import cv2
import mediapipe as mp

# Initialize MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Start webcam
cap = cv2.VideoCapture(0)

with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert image color (BGR -> RGB)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Detect pose
        results = pose.process(image)

        # Convert back to BGR for OpenCV
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            # Draw only arm nodes (shoulders, elbows, wrists)
            landmarks = results.pose_landmarks.landmark
            h, w, _ = image.shape
            arm_points = [11, 13, 15, 12, 14, 16]  # L/R shoulders, elbows, wrists
            for i in arm_points:
                x = int(landmarks[i].x * w)
                y = int(landmarks[i].y * h)
                cv2.circle(image, (x, y), 10, (0, 255, 0), -1)

        cv2.imshow('Arm Node Detection', image)

        if cv2.waitKey(5) & 0xFF == 27:  # ESC to quit
            break

cap.release()
cv2.destroyAllWindows()
