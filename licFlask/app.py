import os
from flask import Flask, render_template, request, send_from_directory
import cv2
import numpy as np
import pytesseract
from PIL import Image

# Path to the directory containing Tesseract-OCR data files
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'


app = Flask(__name__)

# Path to the haarcascade file
CASCADE_PATH = "Face_cascade.xml"

# Load the face cascade
FACE_CASCADE = cv2.CascadeClassifier(CASCADE_PATH)

# Create the uploads directory if it doesn't exist
UPLOADS_DIR = 'uploads'
if not os.path.exists(UPLOADS_DIR):
    os.makedirs(UPLOADS_DIR)

# Function to detect faces in the image
def detect_faces(image_path, filename, display=True):
    image = cv2.imread(image_path)
    image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(image_grey, scaleFactor=1.16, minNeighbors=5, minSize=(25, 25), flags=0)

    extracted_dir = 'Extracted'
    if not os.path.exists(extracted_dir):
        os.makedirs(extracted_dir)

    for existing_image in os.listdir(extracted_dir):
        os.remove(os.path.join(extracted_dir, existing_image))
        
    for x, y, w, h in faces:
        sub_img = image[y-10:y+h+10, x-10:x+w+10]
        cv2.imwrite(f"{extracted_dir}/{filename}.jpg", sub_img)
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 0), 2)

    if display:
        cv2.imshow("Faces Found", image)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')

        file = request.files['file']

        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return render_template('index.html', error='No selected file')

        if file:
            # Save the uploaded image
            image_path = os.path.join(UPLOADS_DIR, file.filename)
            file.save(image_path)

            # Detect faces in the image
            detect_faces(image_path, "uploaded_image", display=False)

            extracted_face_path = os.path.join('Extracted', 'uploaded_image.jpg')

            # Pass the results to the template
            extracted_text_img = image_path # = pytesseract.image_to_string(Image.open(image_path))
            print('taking image from ', image_path)
            text = pytesseract.image_to_string(extracted_text_img, config='--tessdata-dir ./tessdata_best/')
            print('TEXT ', text)
            

            lic_text = ""
            dob_text = ""

            for line in text.split('\n'):
                if 'license' in line.lower():
                    lic_text = line.replace('License No. : ', '')[:16]
                if 'dob' in line.lower():
                    if 'DOB : ' in line:
                        dob_text = line.replace('DOB : ', '')[:10]
                    else:
                        dob_text = line.replace('DOB ', '')[:10]

                    
                
            return render_template('index.html',image_file=image_path, image_file_face=extracted_face_path, license_text=lic_text, dob_text=dob_text)

    return render_template('index.html')



@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOADS_DIR, filename)

EXTRACTED_DIR = 'Extracted'

# Route to serve extracted images
@app.route('/Extracted/<path:filename>')
def extracted_file(filename):
    return send_from_directory(EXTRACTED_DIR, filename)

if __name__ == '__main__':
    app.run(debug=True)
