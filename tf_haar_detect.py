from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
import cv2
from ultralytics import YOLO

model_path = r"D:\ML_Projects\Drowsiness-Detection-System\Models\drowse_model_tf_2.h5"
model = load_model(model_path)
#face_model = YOLO(r"D:\ML_Projects\Drowsiness-Detection-System\Models\yolov8n-face.pt")

eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def preprocess_image(img_array):
    img_array = cv2.resize(img_array, (256, 256))
    img_array = img_array.astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

vid = cv2.VideoCapture(0)
while vid.isOpened():
    ret, frame = vid.read()

    if not ret:
        break

    eyes = eye_cascade.detectMultiScale(gray, 1.1, 4)
    for(x, y, w, h) in eyes:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    
        prediction = model.predict(preprocessed_frame)
    
    if prediction[0][0] < 0.5:
        msg = "Drowsy"
    else:
        msg = "Not Drowsy"

        cv2.putText(frame, msg, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Detected Face", frame)

    key = cv2.waitKey(1)
    if key == ord("e"):
        break

vid.release()
cv2.destroyAllWindows()