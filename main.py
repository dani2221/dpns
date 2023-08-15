import cv2
from mtcnn import MTCNN
from yoloface import face_analysis
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import base64
import uuid
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static'
CORS(app)  # Enable Cross-Origin Resource Sharing

face=face_analysis()
face_cascade = cv2.CascadeClassifier('./configs/haarcascade_frontalface_default.xml')
detector = MTCNN()


def blur_faces1(image, pixelate=False):
    tm = time.time()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        if pixelate:
            face = image[y:y + h, x:x + w]
            face = cv2.resize(face, (10, 10), interpolation=cv2.INTER_NEAREST)
            image[y:y + h, x:x + w] = cv2.resize(face, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            face = image[y:y + h, x:x + w]
            blurred_face = cv2.GaussianBlur(face, (99, 99), 15)  # You can adjust blur parameters
            image[y:y + h, x:x + w] = blurred_face

    print(time.time()-tm)
    return image


def blur_faces2(image, pixelate=False):
    tm = time.time()

    faces = detector.detect_faces(image)

    for face in faces:
        x, y, w, h = face['box']
        blurred_face = cv2.GaussianBlur(image[y:y + h, x:x + w], (99, 99), 30)  # You can adjust blur parameters
        image[y:y + h, x:x + w] = blurred_face

    print(time.time() - tm)
    return image


def yolo_face_detection(image, confidence_threshold=0.5, nms_threshold=0.3, pixelate=False):
    tm = time.time()
    img,box,conf=face.face_detection(image, model='tiny')

    for i in range(len(box)):
        x, y, h, w = box[i]
        blurred_face = cv2.GaussianBlur(img[y:y + h, x:x + w], (99, 99), 30)  # You can adjust blur parameters
        img[y:y + h, x:x + w] = blurred_face

    print(time.time() - tm)
    return img

@app.route('/blur', methods=['POST'])
def blur():
    image_file = request.files['imagefile']

    # Get JSON parameters from request
    pixelate = request.form.get('pixelate')
    is_video = request.form.get('is_video')
    method = request.form.get('method')
    imagefile = request.files.get('imagefile', '')

    image_np = np.fromstring(imagefile.read(), np.uint8)
    image_cv2 = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    if method == '0':
        result = blur_faces1(image_cv2, pixelate)
    elif method == '1':
        result = blur_faces2(image_cv2, pixelate)
    elif method == '2':
        result = yolo_face_detection(image_cv2)

    _, processed_image_encoded = cv2.imencode('.jpg', image_cv2)
    path = os.path.join(app.config['UPLOAD_FOLDER'], str(uuid.uuid4())+'.jpg')
    with open(path, 'wb') as f:
        f.write(processed_image_encoded.tobytes())

    # Prepare a response
    response_data = {
        'processed_image': path
    }

    return jsonify(response_data)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
image_path = 'posters/hw7_poster_1 (1).jpg'
yolo_face_detection(image_path, pixelate=True)
