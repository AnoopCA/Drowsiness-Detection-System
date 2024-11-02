from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
import cv2
from ultralytics import YOLO

model_path = r"D:\ML_Projects\Drowsiness-Detection-System\Models\drowse_model_tf_1.h5"
model = load_model(model_path)
face_model = YOLO(r"D:\ML_Projects\Drowsiness-Detection-System\Models\yolov8n-face.pt")

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
    results = face_model(frame)
    boxes = results[0].boxes

    for box in boxes:
        top_left_x = int(box.xyxy.tolist()[0][0])
        top_left_y = int(box.xyxy.tolist()[0][1])
        bottom_right_x = int(box.xyxy.tolist()[0][2])
        bottom_right_y = int(box.xyxy.tolist()[0][3])

        face_img = frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

        # -------------------------- Add new code to detect drowsiness using Eye Aspect Ratio -------------------------- #
        
        preprocessed_frame = preprocess_image(face_img)
        prediction = model.predict(preprocessed_frame)
        if prediction[0][0] < 0.5:
            msg = "Drowsy"
        else:
            msg = "Not Drowsy"

        # ------------------------- Drowsiness detection section end here --------------------------------------------- #

        cv2.putText(frame, msg, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Detected Face", frame)

    key = cv2.waitKey(1)
    if key == ord("e"):
        break

vid.release()
cv2.destroyAllWindows()