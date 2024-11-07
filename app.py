import streamlit as st
import numpy as np
import cv2
from keras.models import load_model
from keras.utils import load_img, img_to_array
import tempfile

model_path = r"D:\ML_Projects\Drowsiness-Detection-System\Models\drowse_model_tf_7_epoch_250.h5"
drowse_model = load_model(model_path)
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def preprocess_image(img_array):
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    img_array = cv2.resize(img_array, (80, 80))
    img_array = img_array.astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

st.title("Drowsiness Detection System")
choice = st.sidebar.selectbox("MENU", ("HOME", "IMAGE", "VIDEO", "CAMERA"))

if choice == "HOME":
    st.header("Welcome!")

elif choice == "IMAGE":
    file = st.file_uploader("Upload Image")
    if file:
        b = file.getvalue()
        d = np.frombuffer(b, np.uint8)
        img = cv2.imdecode(d, cv2.IMREAD_COLOR)
        eyes = eye_cascade.detectMultiScale(img)
        for (x,y,l,w) in eyes:
            crop_eye = img[y:y+w, x:x+l]
            crop_eye = img_to_array(crop_eye)
            crop_eye = preprocess_image(crop_eye)
            pred = drowse_model.predict(crop_eye)[0][0]
            if pred < 0.5:
                cv2.rectangle(img, (x,y), (x+l,y+w), (0,0,255), 2)
                cv2.putText(img, "Drowsy", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            else:
                cv2.rectangle(img, (x,y), (x+l,y+w), (0,255,0), 2)
                cv2.putText(img, "Not Drowsy", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        st.image(img, channels='BGR')

elif choice == "VIDEO":
    file = st.file_uploader("Upload Video")
    windows = st.empty()
    if file:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(file.read())
        vid = cv2.VideoCapture(tmp_file.name)
        while(vid.isOpened()):
            flag, frame=vid.read()
            if (flag):
                eyes = eye_cascade.detectMultiScale(frame)
                for (x,y,l,w) in eyes:
                    crop_eye = frame[y:y+w, x:x+l]
                    crop_eye = img_to_array(crop_eye)
                    crop_eye = preprocess_image(crop_eye)
                    pred = drowse_model.predict(crop_eye)[0][0]
                    if pred < 0.5:
                        cv2.rectangle(frame, (x,y), (x+l,y+w), (0,0,255), 2)
                        cv2.putText(frame, "Drowsy", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    else:
                        cv2.rectangle(frame, (x,y), (x+l,y+w), (0,255, 0), 2)
                        cv2.putText(frame, "Not Drowsy", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                    windows.image(frame, channels="BGR")

elif choice == "CAMERA":
    st.session_state["CAMERA"] = True
    k = st.text_input("Enter 0 to open webcam or write URL for opening IP camera")
    if len(k) == 1:
        k = int(k)
    btn = st.button("Start Camera")
    cls_btn = st.button("Stop Camera")
    if cls_btn:
        st.session_state["CAMERA"] = False
        vid.release()
        cv2.destroyAllWindows()

    windows = st.empty()
    if btn and st.session_state["CAMERA"]:
        vid = cv2.VideoCapture(k)
        while(vid.isOpened()):
            flag, frame=vid.read()
            if (flag):
                eyes = eye_cascade.detectMultiScale(frame)
                for (x,y,l,w) in eyes:
                    crop_eye = frame[y:y+w, x:x+l]
                    crop_eye = img_to_array(crop_eye)
                    crop_eye = preprocess_image(crop_eye)
                    pred = drowse_model.predict(crop_eye)[0][0]
                    if pred < 0.5:
                        cv2.rectangle(frame, (x,y), (x+l,y+w), (0,0,255), 2)
                        cv2.putText(frame, "Drowsy", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    else:
                        cv2.rectangle(frame, (x,y), (x+l,y+w), (0,255,0), 2)
                        cv2.putText(frame, "Not Drowsy", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                    windows.image(frame, channels="BGR")
        vid.release()
        cv2.destroyAllWindows()
