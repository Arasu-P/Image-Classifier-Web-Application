import joblib
import json
import numpy as np
import base64
import cv2
import os
from wavelet import w2d
__class_name_to_number={}
__class_number_to_name={}

__model =None

def classify_image(image_base64_data, file_path=None):
    imgs = get_cropped_image_if_2_eyes(file_path, image_base64_data)
    result = []

    for img in imgs:
        if img is None:
            continue

        scaled_raw_img = cv2.resize(img, (32, 32))
        img_har = w2d(img, 'db1', 5)
        scaled_img_har = cv2.resize(img_har, (32, 32))

        combined_img = np.vstack((
            scaled_raw_img.reshape(32*32*3, 1),
            scaled_img_har.reshape(32*32, 1)
        ))

        final = combined_img.reshape(1, 32*32*3 + 32*32).astype(float)

        # predicted class name (string)
        predicted_name = str(__model.predict(final)[0])

        # probabilities for ALL classes (same order as model.classes_)
        probs = __model.predict_proba(final)[0]
        probs_percent = [round(float(p * 100), 2) for p in probs]

        result.append({
            'class': predicted_name,
            'class_probability': probs_percent,
            'class_dictionary': __class_name_to_number
        })

    return result


def load_saved_artifacts():
    print("loading saved artifacts...start")
    global __class_name_to_number
    global __class_number_to_name

    with open(r"C:\Users\ARASU P\Image Classifier Project\server\artifacts\class_dictionary.json", "r") as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v:k for k,v in __class_name_to_number.items()}

    global __model
    if __model is None:
        with open(r"C:\Users\ARASU P\Image Classifier Project\server\artifacts\saved_model.pkl", "rb") as f:
            __model = joblib.load(f)
    print("loading saved artifacts...done")

def class_number_to_name(class_num):
    return __class_number_to_name[class_num]

def get_cv2_image_from_base64_string(b64str):
    '''
    credit: https://stackoverflow.com/questions/33754935/read-a-base-64-encoded-image-from-memory-using-opencv-python-library
    :param uri:
    :return:
    '''
    encoded_data = b64str.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def get_cropped_image_if_2_eyes(image_path, image_base64_data):
    face_cascade = cv2.CascadeClassifier(
    r"C:\Users\ARASU P\Image Classifier Project\server\opencv\haarcascades\haarcascade_frontalface_default.xml"
    )

    eye_cascade = cv2.CascadeClassifier(
      r"C:\Users\ARASU P\Image Classifier Project\server\opencv\haarcascades\haarcascade_eye.xml"
     )

    if image_path:
        img = cv2.imread(os.path.join(os.path.dirname(__file__), image_path))
    else:
        img = get_cv2_image_from_base64_string(image_base64_data)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    cropped_faces = []
    for (x,y,w,h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            if len(eyes) >= 2:
                cropped_faces.append(roi_color)
    return cropped_faces

def get_b64_test_image_for_sam():
    with open(r"C:\Users\ARASU P\Image Classifier Project\server\b64.txt") as f:
        return f.read()

if __name__ =='__main__': 
    load_saved_artifacts()
    #print(classify_image(get_b64_test_image_for_sam(),None))
    print(classify_image(None, "./test_images/images 1.jpg"))
    print(classify_image(None, "./test_images/images 2.jpg"))
    print(classify_image(None, "./test_images/images 3.jpg"))
    print(classify_image(None, "./test_images/images 4.jpg"))
    print(classify_image(None, "./test_images/images 5.jpg")) 
    print(classify_image(None, "./test_images/images 6.jpg"))
    print(classify_image(None, "./test_images/images 7.jpg"))
    print(classify_image(None, "./test_images/images 8.jpg"))
    print(classify_image(None, "./test_images/images 9.jpg"))


