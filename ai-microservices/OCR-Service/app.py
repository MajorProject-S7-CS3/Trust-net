import pytesseract
from flask import Flask, request, jsonify
from PIL import Image, ImageEnhance, ImageFilter
import re

app = Flask(__name__)

@app.route('/ocr', methods=['POST'])
def ocr():
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file part"}), 400

    file = request.files['file']
    image = Image.open(file.stream)

    # Preprocessing
    image = image.convert('L')  # Grayscale
    image = image.filter(ImageFilter.SHARPEN)
    image = image.point(lambda x: 0 if x < 140 else 255)  # Binarize

    custom_config = r'--oem 3 --psm 6'
    raw_text = pytesseract.image_to_string(image, config=custom_config)

    # Aadhaar Pattern: 4 digits + space + 4 digits + space + 4 digits
    aadhaar_match = re.search(r'\b\d{4}\s\d{4}\s\d{4}\b', raw_text)
    aadhaar_number = aadhaar_match.group() if aadhaar_match else None

    # Name (assumed to be first line after header)
    lines = raw_text.strip().split('\n')
    possible_names = [line.strip() for line in lines if re.match(r'^[A-Z][a-z]+\s[A-Z][a-z]+$', line.strip())]
    name = possible_names[0] if possible_names else None
    # DOB
    dob_match = re.search(r'(\d{2}/\d{2}/\d{4}|\d{2}-\d{2}-\d{4}|\d{2}\s?\w+\s?\d{4})', raw_text)
    dob = dob_match.group() if dob_match else None

    # Gender
    gender_match = re.search(r'\b(MALE|FEMALE|OTHER)\b', raw_text, re.IGNORECASE)
    gender = gender_match.group().upper() if gender_match else None

    return jsonify({
        "status": "success",
        "extracted_text": raw_text,
        "aadhaar_number": aadhaar_number,
        "name": name,
        "dob": dob,
        "gender": gender
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
