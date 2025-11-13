import cv2
from ultralytics import YOLO

# Load YOLOv8 Pose model
model = YOLO("yolov8s-pose.pt")

# Start webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)

    # store per-person x-coordinate lists
    person_x_coords = []

    for r in results:
        if r.keypoints is None:
            continue

        keypoints = r.keypoints.xy  # shape: (num_people, 17, 2)

        for person_idx, person in enumerate(keypoints):
            x_coords = []  # all x-values for this person

            # Collect coordinates for arm keypoints
            indices = [5, 6, 7, 8, 9, 10]  # L/R shoulder, elbow, wrist
            for i in indices:
                x = float(person[i][0])
                x_coords.append(x)

            # Save this person's sorted list
            person_x_coords.append(sorted(x_coords))

            # Convert to int tuples for drawing
            L_shoulder = tuple(person[5].int().tolist())
            L_elbow    = tuple(person[7].int().tolist())
            L_wrist    = tuple(person[9].int().tolist())
            R_shoulder = tuple(person[6].int().tolist())
            R_elbow    = tuple(person[8].int().tolist())
            R_wrist    = tuple(person[10].int().tolist())

            # Midpoint between shoulders
            mid_shoulder = (
                int((L_shoulder[0] + R_shoulder[0]) / 2),
                int((L_shoulder[1] + R_shoulder[1]) / 2)
            )

            # Draw mid-shoulder node
            cv2.circle(frame, mid_shoulder, 10, (0, 255, 255), -1)

            # Draw arm lines
            cv2.line(frame, mid_shoulder, L_elbow, (255, 0, 0), 3)
            cv2.line(frame, L_elbow, L_wrist, (255, 0, 0), 3)
            cv2.line(frame, mid_shoulder, R_elbow, (255, 0, 0), 3)
            cv2.line(frame, R_elbow, R_wrist, (255, 0, 0), 3)

            # Draw joint nodes
            for point in [L_elbow, L_wrist, R_elbow, R_wrist]:
                cv2.circle(frame, point, 6, (0, 255, 0), -1)

    # Print each person's x-coordinate list
    for i, coords in enumerate(person_x_coords, start=1):
        print(f"Person {i} sorted x-coordinates: {coords}")

    # Display the camera feed
    cv2.imshow("Multi-Person Arm Node Detection", frame)

    if cv2.waitKey(5) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
