from flask import Flask, request, jsonify
from deepface import DeepFace
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/face-match', methods=['POST'])
def face_match():
    if 'selfie' not in request.files or 'document' not in request.files:
        return jsonify({'error': 'Both selfie and document images are required'}), 400

    selfie_file = request.files['selfie']
    document_file = request.files['document']

    selfie_filename = secure_filename(selfie_file.filename)
    document_filename = secure_filename(document_file.filename)

    selfie_path = os.path.join(UPLOAD_FOLDER, selfie_filename)
    document_path = os.path.join(UPLOAD_FOLDER, document_filename)

    # Save the uploaded images
    selfie_file.save(selfie_path)
    document_file.save(document_path)

    # Optional: Model can be passed in form-data
    model_name = request.form.get('model', 'Facenet512')  # Default to Facenet512

    try:
        result = DeepFace.verify(img1_path=document_path, img2_path=selfie_path, model_name=model_name)

        
        distance = result["distance"]
        threshold = 0.55  # Customize this based on your model; Facenet512 default is ~0.3–0.4
        match = distance < threshold
        confidence = round((1 - distance) * 100, 2)

        response = {
            "match": match,
            "confidence": confidence,
            "distance": distance,
            "threshold": threshold
        }

        return jsonify(response)

    except ValueError as ve:
        return jsonify({'error': f'Face detection error: {str(ve)}'}), 400

    except Exception as e:
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

    finally:
        # Optional cleanup
        if os.path.exists(selfie_path):
            os.remove(selfie_path)
        if os.path.exists(document_path):
            os.remove(document_path)

if __name__ == '__main__':
    app.run(debug=True)
