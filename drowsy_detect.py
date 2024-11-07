from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
import cv2

SKIP_FRAMES_BY = 15
FRAMES_THRSH = 3
count = 0
eye_count = 0
eye_closed = 0

#model_path = r"D:\ML_Projects\Drowsiness-Detection-System\Models\drowse_model_tf_2.h5"
model_path = r"D:\ML_Projects\Drowsiness-Detection-System\Models\drowse_model_tf_7_epoch_250.h5"

model = load_model(model_path)
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def preprocess_image(img_array):
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    img_array = cv2.resize(img_array, (80, 80))
    img_array = img_array.astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

vid = cv2.VideoCapture(0) #r"D:\ML_Projects\Drowsiness-Detection-System\Data\anp_test.mp4")
while vid.isOpened():
    ret, frame = vid.read()
    if not ret:
        break
    count += 1
    score = 0
    if (count % SKIP_FRAMES_BY) == 0:
        eyes = eye_cascade.detectMultiScale(frame, 1.1, 9)
        for(x, y, w, h) in eyes:
            eye_count += 1
            img = frame[y:y+h, x:x+w]
            #cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            pred = model.predict(preprocess_image(img))
            score += pred[0][0]
        if eye_count == 0:
            continue
        else:
            score = score / eye_count
            eye_count = 0
        if score < 0.7:
            if eye_closed < (FRAMES_THRSH+1):
                eye_closed += 1
        else:
            eye_closed -= 1
            if eye_closed < 0:
                eye_closed = 0

    if eye_closed > (FRAMES_THRSH-1):
        cv2.putText(frame, "Drowsy", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "Not Drowsy", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) != -1:
        break

vid.release()
cv2.destroyAllWindows()