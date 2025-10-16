import cv2
import pytesseract
import re
import spacy
import numpy as np
import json
import os

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Predefined constants
UNIVERSITY_NAME = "SCMS SCHOOL OF ENGINEERING AND TECHNOLOGY"
UNI_ADDR = "Vidya Nagar, Karukutty P.O., Aluva, Kerala - 683 576"
UNI_PHONE = "0484 2450330, 2451907"
COURSE_NAME = "B.Tech - Computer Science & Engineering"

# ----------------- Helper Functions -----------------

def preprocess_image(img):
    if img is None:
        raise ValueError("Empty image passed to preprocess_image")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(thresh)
    return enhanced

def correct_ocr_errors(text):
    return (text.replace('O', '0').replace('o', '0')
                .replace('I', '1').replace('l', '1')
                .replace('|', '1').replace('-', '/')
                .replace('\\', '/').replace('_', '/')
                .replace(' / ', '/').replace('/ ', '/')
                .replace(' /', '/'))

def extract_reg_number(text):
    text = correct_ocr_errors(text)
    pattern = r'S[A-Z]{2}/\d{4,5}/\d{2}'
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(0).upper() if match else None

def parse_student_id(img):
    result = {
        "University": UNIVERSITY_NAME,
        "University Address": UNI_ADDR,
        "University Phone": UNI_PHONE,
        "Name": None,
        "Registration Number": None,
        "DOB": None,
        "Course": COURSE_NAME
    }

    pre_img = preprocess_image(img)
    raw_text = pytesseract.image_to_string(pre_img, config="--psm 6")
    lines = [line.strip() for line in raw_text.split("\n") if line.strip()]
    text_joined = " ".join(lines)

    # Registration Number
    reg_number = extract_reg_number(text_joined)
    result["Registration Number"] = reg_number

    # DOB
    dob_match = re.search(r'\b\d{2}/\d{2}/\d{4}\b', text_joined)
    if dob_match:
        result["DOB"] = dob_match.group(0)

    # Name detection using SpaCy NER
    doc = nlp(text_joined)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            result["Name"] = ent.text
            break

    # Fallback: look for line starting with "Name"
    if not result["Name"]:
        for line in lines:
            if re.search(r'\bName\b', line, re.IGNORECASE):
                parts = line.split(':')
                if len(parts) > 1:
                    result["Name"] = parts[-1].strip()
                else:
                    result["Name"] = re.sub(r'(?i).*name', '', line).strip()
                break

    return result

# ----------------- Main Script -----------------

if __name__ == "__main__":
    # Folder containing sample images
    folder_path = r"C:\Reshma\Trustnet\ai-microservices\OCR-Service\sample_images"

    # List all images in folder (jpg, png)
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)
        img = cv2.imread(img_path)

        if img is None:
            print(f"[ERROR] Cannot read image: {img_path}")
            continue

        print(f"\n--- Results for {img_file} ---")
        result = parse_student_id(img)
        print(json.dumps(result, indent=4))
