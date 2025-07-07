from flask import Flask, request, jsonify
import face_recognition
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

@app.route('/verify-face', methods=['POST'])
def verify_face():
    if 'document' not in request.files or 'selfie' not in request.files:
        return jsonify({'status': 'error', 'message': 'Missing document or selfie'}), 400

    doc_img = face_recognition.load_image_file(request.files['document'])
    selfie_img = face_recognition.load_image_file(request.files['selfie'])

    try:
        doc_enc = face_recognition.face_encodings(doc_img)[0]
        selfie_enc = face_recognition.face_encodings(selfie_img)[0]

        match = face_recognition.compare_faces([doc_enc], selfie_enc)[0]
        distance = face_recognition.face_distance([doc_enc], selfie_enc)[0]

        return jsonify({
            'status': 'success',
            'match': match,
            'confidence': float(1 - distance)
        })

    except IndexError:
        return jsonify({'status': 'error', 'message': 'No face detected in one of the images'}), 400

if __name__ == '__main__':
    app.run(port=5001, debug=True)
