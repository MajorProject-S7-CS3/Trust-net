from flask import Flask, jsonify, request
import cv2
import mediapipe as mp
import time
import requests

app = Flask(__name__)

# Mediapipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

def get_ear(landmarks, eye_indices, frame_width, frame_height):
    points = [(int(landmarks[i].x * frame_width), int(landmarks[i].y * frame_height)) for i in eye_indices]
    horizontal = ((points[0][0] - points[3][0]) ** 2 + (points[0][1] - points[3][1]) ** 2) ** 0.5
    vertical1 = ((points[1][0] - points[5][0]) ** 2 + (points[1][1] - points[5][1]) ** 2) ** 0.5
    vertical2 = ((points[2][0] - points[4][0]) ** 2 + (points[2][1] - points[4][1]) ** 2) ** 0.5
    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return ear

@app.route('/liveness-check', methods=['POST'])
def liveness_check():
    blink_count = 0
    blink_threshold = 5
    ear_threshold = 0.22

    cap = cv2.VideoCapture(0)
    captured_image = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            left_ear = get_ear(landmarks, LEFT_EYE, w, h)
            right_ear = get_ear(landmarks, RIGHT_EYE, w, h)

            if left_ear < ear_threshold and right_ear < ear_threshold:
                blink_count += 1
                print(f"Blink #{blink_count}")
                time.sleep(0.2)

        if blink_count >= blink_threshold:
            ret, jpeg = cv2.imencode('.jpg', frame)
            captured_image = jpeg.tobytes()
            break

        cv2.imshow("Blink Detection", frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    if not captured_image:
        return jsonify({"error": "Blink detection failed"}), 400

    doc_file = request.files.get('document')
    if not doc_file:
        return jsonify({"error": "No document file sent"}), 400

    files = {
        "document": (doc_file.filename, doc_file.stream, doc_file.mimetype),
        "selfie": ("captured.jpg", captured_image, 'image/jpeg')
    }

    try:
        response = requests.post("http://localhost:5000/face-match", files=files)
        return jsonify(response.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/")
def home():
    return "Liveness Check API is running."

if __name__ == '__main__':
    app.run(port=5002)