import base64
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from random import randint
import cv2
import sys
import os
import traceback

import pytesseract as tess
tess.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
from PIL import Image

app = Flask(__name__)

CASCADE = "Face_cascade.xml"
FACE_CASCADE = cv2.CascadeClassifier(CASCADE)

def detect_faces(image_path, filename, display=True):
    image=cv2.imread(image_path)
    image_grey=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(image_grey,scaleFactor=1.16,minNeighbors=5,minSize=(25,25),flags=0)

    

    existing_images = os.listdir("Extracted")
    for existing_image in existing_images:
        os.remove(os.path.join("Extracted", existing_image))
        
    for x,y,w,h in faces:
        sub_img=image[y-10:y+h+10,x-10:x+w+10]
        # random_name = str(randint(0, 10000))
        # base64_name = base64.b64encode(random_name.encode()).decode('utf-8')
        cv2.imwrite(f"Extracted/{filename}.jpg", sub_img)
        os.chdir("../")
        cv2.rectangle(image,(x,y),(x+w,y+h),(255, 255,0),2)

    if display:
            cv2.imshow("Faces Found",image)
            # if (cv2.waitKey(0) & 0xFF == ord('q')) or (cv2.waitKey(0) & 0xFF == ord('Q')):
            # 	cv2.destroyAllWindows()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect_face', methods=['POST'])
def detect_face():
    if 'image' not in request.files:
        return jsonify({'success': False, 'message': 'No file part'})

    file = request.files['image']

    if file.filename == '':
        return jsonify({'success': False, 'message': 'No selected file'})

    try:
        # Delete the existing "uploads" folder and its contents
        if os.path.exists("uploads"):
            os.system("rmdir /s /q uploads")

        # Create a new "uploads" folder
        os.mkdir("./uploads")
        
        # Delete the existing "Extracted" folder and its contents
        if os.path.exists("Extracted"):
            os.system("rmdir /s /q Extracted")

        # Create a new "Extracted" folder
        os.mkdir("./Extracted")

        filename = secure_filename(file.filename)
        filepath = os.path.join("uploads", filename)
        file.save(filepath)

        detect_faces(filepath, filename[:-4], False)
        
        img = Image.open(filepath)
        text = tess.image_to_string(img)
        print(text)

        # Assuming your detect_faces function saves the face image
        face_image_path = os.path.join("Extracted", f"{filename[:-4]}.jpg")
        return jsonify({'success': True, 'message': 'Face detected successfully', 'face_image_path': face_image_path, 'extracted_text': text})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'message': 'Error processing the image'})


if __name__ == '__main__':
    # Delete the existing "uploads" folder and its contents
    if os.path.exists("uploads"):
        import shutil
        shutil.rmtree("uploads")

    # Create a new "uploads" folder
    os.mkdir("uploads")

    # Run the Flask app
    app.run(debug=True)
