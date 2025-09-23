import cv2
import pytesseract
from PIL import Image
import re
import numpy as np

# Preprocess image to improve OCR
def preprocess_image(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Adaptive thresholding to enhance faded text
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    # Optional: increase contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(thresh)
    return enhanced

# Extract red text specifically
def extract_red_text(image_path):
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define range for red color in HSV
    # Red has two ranges in HSV (due to hue wrapping)
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    
    # Create masks for red color
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    
    # Apply morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    
    # Create red text image
    red_text_img = cv2.bitwise_and(img, img, mask=red_mask)
    
    # Convert to grayscale for OCR
    red_gray = cv2.cvtColor(red_text_img, cv2.COLOR_BGR2GRAY)
    
    # Enhance red text contrast
    _, red_binary = cv2.threshold(red_gray, 1, 255, cv2.THRESH_BINARY)
    
    # Apply additional enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced_red = clahe.apply(red_binary)
    
    return enhanced_red, red_mask

# Get bounding boxes of red text regions
def get_red_text_regions(red_mask):
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = []
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Filter out very small regions (noise)
        if w > 20 and h > 10:
            regions.append({
                'x': x,
                'y': y,
                'w': w,
                'h': h,
                'area': w * h,
                'center_y': y + h//2
            })
    
    # Sort regions by vertical position (top to bottom)
    regions.sort(key=lambda r: r['center_y'])
    return regions

# Extract text from specific regions
def extract_text_from_regions(img, regions):
    texts = []
    custom_config = r'--oem 3 --psm 7'
    
    for region in regions:
        # Extract region with some padding
        x, y, w, h = region['x'], region['y'], region['w'], region['h']
        padding = 5
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(img.shape[1], x + w + padding)
        y_end = min(img.shape[0], y + h + padding)
        
        roi = img[y_start:y_end, x_start:x_end]
        
        if roi.size > 0:
            text = pytesseract.image_to_string(roi, config=custom_config).strip()
            if text:  # Only add non-empty text
                texts.append({
                    'text': text,
                    'region': region
                })
    
    return texts

# Correct common OCR misreads specifically for registration numbers
def correct_ocr_errors(text):
    corrections = text
    # Common OCR mistakes for registration numbers
    corrections = corrections.replace('O', '0')  # O to 0
    corrections = corrections.replace('o', '0')  # lowercase o to 0
    corrections = corrections.replace('I', '1')  # I to 1
    corrections = corrections.replace('l', '1')  # l to 1
    corrections = corrections.replace('|', '1')  # | to 1
    corrections = corrections.replace('-', '/')  # - to /
    corrections = corrections.replace('\\', '/') # \ to /
    corrections = corrections.replace('_', '/')  # _ to /
    corrections = corrections.replace(' / ', '/')  # Remove spaces around /
    corrections = corrections.replace('/ ', '/')   # Remove space after /
    corrections = corrections.replace(' /', '/')   # Remove space before /
    
    return corrections

# Extract registration number with enhanced pattern matching
def extract_reg_number(text):
    text = correct_ocr_errors(text)
    
    # Primary pattern: S + 2 letters + / + 4-5 digits + / + 2 digits
    patterns = [
        r'\bS[A-Z]{2}/\d{4,5}/\d{2}\b',           # Standard format
        r'\bS[A-Z]{2}\s*/\s*\d{4,5}\s*/\s*\d{2}\b',  # With possible spaces
        r'S[A-Z]{2}/\d{4,5}/\d{2}',               # Without word boundaries
        r'S[a-zA-Z]{2}/\d{4,5}/\d{2}',            # Case insensitive letters
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            # Clean up the match (remove extra spaces, convert to uppercase)
            clean_match = re.sub(r'\s+', '', match.group(0)).upper()
            # Validate the cleaned format
            if re.match(r'^S[A-Z]{2}/\d{4,5}/\d{2}

# Parse student ID image with red text detection
def parse_student_id(image_path):
    # Extract red text
    red_enhanced, red_mask = extract_red_text(image_path)
    red_regions = get_red_text_regions(red_mask)
    red_texts = extract_text_from_regions(red_enhanced, red_regions)
    
    # Assign university and course from red text
    university = None
    course = None
    
    if len(red_texts) >= 1:
        university = red_texts[0]['text']  # First red text as university
    if len(red_texts) >= 2:
        course = red_texts[1]['text']      # Second red text as course
    
    # Preprocess for general OCR (registration number, DOB, etc.)
    preprocessed_img = preprocess_image(image_path)
    
    # OCR with config: alphanumeric + slash only
    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/'
    text_data = pytesseract.image_to_data(preprocessed_img, output_type=pytesseract.Output.DICT, config=custom_config)
    
    lines = []
    for i, word in enumerate(text_data['text']):
        if word.strip() != "":
            lines.append({
                "text": word,
                "left": text_data['left'][i],
                "top": text_data['top'][i],
                "height": text_data['height'][i]
            })
    
    # Filter out principal/signature
    filtered_lines = [line for line in lines if "principal" not in line['text'].lower()]
    
    # Combine text for regex search
    full_text = " ".join([line['text'] for line in filtered_lines])
    
    # Extract registration number
    reg_number = extract_reg_number(full_text)
    
    # Simple DOB detection (DD/MM/YYYY)
    dob_match = re.search(r'\b\d{2}/\d{2}/\d{4}\b', full_text)
    dob = dob_match.group(0) if dob_match else None
    
    return {
        "Registration Number": reg_number,
        "DOB": dob,
        "University": university,
        "Course": course,
        "Red Text Detected": len(red_texts),
        "All Red Texts": [rt['text'] for rt in red_texts]  # For debugging
    }

# Enhanced version with additional red color detection for different lighting
def extract_red_text_enhanced(image_path):
    img = cv2.imread(image_path)
    
    # Convert to multiple color spaces for better red detection
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # HSV-based red detection (existing method)
    lower_red1 = np.array([0, 30, 30])
    upper_red1 = np.array([15, 255, 255])
    lower_red2 = np.array([160, 30, 30])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    hsv_red_mask = cv2.bitwise_or(mask1, mask2)
    
    # LAB-based red detection (better for different lighting)
    # In LAB, 'a' channel represents green-red axis
    lower_lab = np.array([0, 130, 0])
    upper_lab = np.array([255, 255, 255])
    lab_red_mask = cv2.inRange(lab, lower_lab, upper_lab)
    
    # Combine both masks
    combined_mask = cv2.bitwise_or(hsv_red_mask, lab_red_mask)
    
    # Apply morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    
    # Create red text image
    red_text_img = cv2.bitwise_and(img, img, mask=combined_mask)
    red_gray = cv2.cvtColor(red_text_img, cv2.COLOR_BGR2GRAY)
    _, red_binary = cv2.threshold(red_gray, 1, 255, cv2.THRESH_BINARY)
    
    return red_binary, combined_mask

# Alternative parser with enhanced red detection
def parse_student_id_enhanced(image_path):
    # Extract red text with enhanced detection
    red_enhanced, red_mask = extract_red_text_enhanced(image_path)
    red_regions = get_red_text_regions(red_mask)
    red_texts = extract_text_from_regions(red_enhanced, red_regions)
    
    # Assign university and course from red text
    university = None
    course = None
    
    if len(red_texts) >= 1:
        university = red_texts[0]['text']  # First red text as university
    if len(red_texts) >= 2:
        course = red_texts[1]['text']      # Second red text as course
    
    # Continue with regular processing for other fields
    preprocessed_img = preprocess_image(image_path)
    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/'
    text_data = pytesseract.image_to_data(preprocessed_img, output_type=pytesseract.Output.DICT, config=custom_config)
    
    lines = []
    for i, word in enumerate(text_data['text']):
        if word.strip() != "":
            lines.append({
                "text": word,
                "left": text_data['left'][i],
                "top": text_data['top'][i],
                "height": text_data['height'][i]
            })
    
    filtered_lines = [line for line in lines if "principal" not in line['text'].lower()]
    full_text = " ".join([line['text'] for line in filtered_lines])
    
    # Collect all text sources for registration number extraction
    all_text_sources = [full_text]
    if red_texts:
        all_text_sources.extend([rt['text'] for rt in red_texts])
    
    # Extract registration number from all sources
    reg_number = extract_reg_from_all_text(all_text_sources)
    
    # Extract DOB
    dob_match = re.search(r'\b\d{2}/\d{2}/\d{4}\b', full_text)
    dob = dob_match.group(0) if dob_match else None
    
    # Also try to find registration number in red text specifically
    reg_from_red = None
    for rt in red_texts:
        temp_reg = extract_reg_number(rt['text'])
        if temp_reg:
            reg_from_red = temp_reg
            break
    
    return {
        "Registration Number": reg_number or reg_from_red,
        "DOB": dob,
        "University": university,
        "Course": course,
        "Red Text Detected": len(red_texts),
        "All Red Texts": [rt['text'] for rt in red_texts],
        "Registration from Red Text": reg_from_red  # For debugging
    }

# Example usage
if __name__ == "__main__":
    image_paths = [
        "/mnt/data/IMG_20250919_105300.jpg",
        "/mnt/data/IMG_20250919_105305.jpg"
    ]
    
    # Use the enhanced version for better red text detection
    results = [parse_student_id_enhanced(path) for path in image_paths]
    
    for i, result in enumerate(results):
        print(f"\n--- Image {i+1} Results ---")
        for key, value in result.items():
            print(f"{key}: {value}"), clean_match):
                return clean_match
    
    return None

# Extract registration number from specific regions (including red text)
def extract_reg_from_all_text(all_texts):
    """
    Extract registration number from combined text sources
    all_texts: list of text strings from different sources
    """
    combined_text = " ".join(all_texts)
    return extract_reg_number(combined_text)

# Parse student ID image with red text detection
def parse_student_id(image_path):
    # Extract red text
    red_enhanced, red_mask = extract_red_text(image_path)
    red_regions = get_red_text_regions(red_mask)
    red_texts = extract_text_from_regions(red_enhanced, red_regions)
    
    # Assign university and course from red text
    university = None
    course = None
    
    if len(red_texts) >= 1:
        university = red_texts[0]['text']  # First red text as university
    if len(red_texts) >= 2:
        course = red_texts[1]['text']      # Second red text as course
    
    # Preprocess for general OCR (registration number, DOB, etc.)
    preprocessed_img = preprocess_image(image_path)
    
    # OCR with config: alphanumeric + slash only
    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/'
    text_data = pytesseract.image_to_data(preprocessed_img, output_type=pytesseract.Output.DICT, config=custom_config)
    
    lines = []
    for i, word in enumerate(text_data['text']):
        if word.strip() != "":
            lines.append({
                "text": word,
                "left": text_data['left'][i],
                "top": text_data['top'][i],
                "height": text_data['height'][i]
            })
    
    # Filter out principal/signature
    filtered_lines = [line for line in lines if "principal" not in line['text'].lower()]
    
    # Combine text for regex search
    full_text = " ".join([line['text'] for line in filtered_lines])
    
    # Extract registration number
    reg_number = extract_reg_number(full_text)
    
    # Simple DOB detection (DD/MM/YYYY)
    dob_match = re.search(r'\b\d{2}/\d{2}/\d{4}\b', full_text)
    dob = dob_match.group(0) if dob_match else None
    
    return {
        "Registration Number": reg_number,
        "DOB": dob,
        "University": university,
        "Course": course,
        "Red Text Detected": len(red_texts),
        "All Red Texts": [rt['text'] for rt in red_texts]  # For debugging
    }

# Enhanced version with additional red color detection for different lighting
def extract_red_text_enhanced(image_path):
    img = cv2.imread(image_path)
    
    # Convert to multiple color spaces for better red detection
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # HSV-based red detection (existing method)
    lower_red1 = np.array([0, 30, 30])
    upper_red1 = np.array([15, 255, 255])
    lower_red2 = np.array([160, 30, 30])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    hsv_red_mask = cv2.bitwise_or(mask1, mask2)
    
    # LAB-based red detection (better for different lighting)
    # In LAB, 'a' channel represents green-red axis
    lower_lab = np.array([0, 130, 0])
    upper_lab = np.array([255, 255, 255])
    lab_red_mask = cv2.inRange(lab, lower_lab, upper_lab)
    
    # Combine both masks
    combined_mask = cv2.bitwise_or(hsv_red_mask, lab_red_mask)
    
    # Apply morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    
    # Create red text image
    red_text_img = cv2.bitwise_and(img, img, mask=combined_mask)
    red_gray = cv2.cvtColor(red_text_img, cv2.COLOR_BGR2GRAY)
    _, red_binary = cv2.threshold(red_gray, 1, 255, cv2.THRESH_BINARY)
    
    return red_binary, combined_mask

# Alternative parser with enhanced red detection
def parse_student_id_enhanced(image_path):
    # Extract red text with enhanced detection
    red_enhanced, red_mask = extract_red_text_enhanced(image_path)
    red_regions = get_red_text_regions(red_mask)
    red_texts = extract_text_from_regions(red_enhanced, red_regions)
    
    # Assign university and course from red text
    university = None
    course = None
    
    if len(red_texts) >= 1:
        university = red_texts[0]['text']  # First red text as university
    if len(red_texts) >= 2:
        course = red_texts[1]['text']      # Second red text as course
    
    # Continue with regular processing for other fields
    preprocessed_img = preprocess_image(image_path)
    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/'
    text_data = pytesseract.image_to_data(preprocessed_img, output_type=pytesseract.Output.DICT, config=custom_config)
    
    lines = []
    for i, word in enumerate(text_data['text']):
        if word.strip() != "":
            lines.append({
                "text": word,
                "left": text_data['left'][i],
                "top": text_data['top'][i],
                "height": text_data['height'][i]
            })
    
    filtered_lines = [line for line in lines if "principal" not in line['text'].lower()]
    full_text = " ".join([line['text'] for line in filtered_lines])
    
    reg_number = extract_reg_number(full_text)
    dob_match = re.search(r'\b\d{2}/\d{2}/\d{4}\b', full_text)
    dob = dob_match.group(0) if dob_match else None
    
    return {
        "Registration Number": reg_number,
        "DOB": dob,
        "University": university,
        "Course": course,
        "Red Text Detected": len(red_texts),
        "All Red Texts": [rt['text'] for rt in red_texts]
    }

# Example usage
if __name__ == "__main__":
    image_paths = [
        "/mnt/data/IMG_20250919_105300.jpg",
        "/mnt/data/IMG_20250919_105305.jpg"
    ]
    
    # Use the enhanced version for better red text detection
    results = [parse_student_id_enhanced(path) for path in image_paths]
    
    for i, result in enumerate(results):
        print(f"\n--- Image {i+1} Results ---")
        for key, value in result.items():
            print(f"{key}: {value}")