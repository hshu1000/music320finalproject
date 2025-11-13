import cv2
from ultralytics import YOLO

# Load YOLOv8 Pose model
model = YOLO("yolov8s-pose.pt")

def detect_arms(frame):
    """
    Runs YOLOv8 pose detection on a given frame and returns the annotated frame
    plus per-person sorted coordinate data.
    """
    results = model(frame, verbose=False)
    annotated = frame.copy()
    person_coords = []

    for r in results:
        if r.keypoints is None:
            continue

        keypoints = r.keypoints.xy
        for person in keypoints:
            x_coords, y_coords = [], []
            indices = [5, 6, 7, 8, 9, 10]  # arm keypoints

            # Collect coordinates
            for i in indices:
                x = float(person[i][0])
                y = float(person[i][1])
                x_coords.append(x)
                y_coords.append(y)

            person_coords.append({
                "x_sorted": sorted(x_coords),
                "y_sorted": sorted(y_coords)
            })

            # Convert to integer tuples
            L_shoulder = tuple(person[5].int().tolist())
            R_shoulder = tuple(person[6].int().tolist())
            L_elbow = tuple(person[7].int().tolist())
            R_elbow = tuple(person[8].int().tolist())
            L_wrist = tuple(person[9].int().tolist())
            R_wrist = tuple(person[10].int().tolist())

            # Midpoint between shoulders
            mid_shoulder = (
                int((L_shoulder[0] + R_shoulder[0]) / 2),
                int((L_shoulder[1] + R_shoulder[1]) / 2)
            )

            # ---- Drawing section ----
            node_color = (60, 255, 100)  # soft green aesthetic
            line_color = (255, 0, 0)     # blue for arm lines
            node_radius = 6
            line_thickness = 3

            # Draw lines for both arms
            cv2.line(annotated, mid_shoulder, L_elbow, line_color, line_thickness)
            cv2.line(annotated, L_elbow, L_wrist, line_color, line_thickness)
            cv2.line(annotated, mid_shoulder, R_elbow, line_color, line_thickness)
            cv2.line(annotated, R_elbow, R_wrist, line_color, line_thickness)

            # Draw nodes (all same color + size)
            for point in [mid_shoulder, L_elbow, L_wrist, R_elbow, R_wrist]:
                cv2.circle(annotated, point, node_radius, node_color, -1)

    return annotated, person_coords


# ---- Optional standalone preview ----
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        annotated, coords = detect_arms(frame)
        cv2.imshow("Pose Detect Test", annotated)
        if cv2.waitKey(5) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
