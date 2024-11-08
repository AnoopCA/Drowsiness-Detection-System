# Import necessary libraries
import streamlit as st
import numpy as np
import cv2
from keras.models import load_model
from keras.utils import load_img, img_to_array
import tempfile

# Load the drowsiness detection model
model_path = r"D:\ML_Projects\Drowsiness-Detection-System\Models\drowse_model_tf_7_epoch_250.h5"
try:
    drowse_model = load_model(model_path)
except:
    pass
# Load the Haar cascade classifier for eye detection
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Function to preprocess images before passing them to the model
def preprocess_image(img_array):
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    img_array = cv2.resize(img_array, (80, 80))
    img_array = img_array.astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Set up the Streamlit app title and menu
st.title("Drowsiness Detection System")
choice = st.sidebar.selectbox("MENU", ("HOME", "IMAGE", "VIDEO", "CAMERA"))

# Display Home Page
if choice == "HOME":
    st.image(r"D:\ML_Projects\Drowsiness-Detection-System\Data\Logo_1.png")
    st.markdown("""The Drowsiness Detection System is a real-time application that monitors and alerts drivers if 
                   signs of drowsiness are detected. It provides a user-friendly menu with options to select different
                   input modes — Image, Video, or Camera, allowing users to test the application’s functionality in 
                   various scenarios. By continuously analyzing eye closure duration, the system detects signs of fatigue 
                   and provides timely visual and audio alerts to promote safer driving and reduce the risk of accidents 
                   caused by drowsiness. This tool is beneficial for drivers, fleet operators, and commercial 
                   transportation companies, offering an automated way to monitor driver alertness.""")

# Image input mode: Upload and process an image to detect drowsiness
elif choice == "IMAGE":
    file = st.file_uploader("Upload Image")
    if file:
        b = file.getvalue()
        # Convert image to NumPy array
        d = np.frombuffer(b, np.uint8)
        # Decode image from buffer
        img = cv2.imdecode(d, cv2.IMREAD_COLOR)
        # Detect eyes in the image
        eyes = eye_cascade.detectMultiScale(img)
        for (x,y,l,w) in eyes:
            # Crop detected eye region
            crop_eye = img[y:y+w, x:x+l]
            # Convert to array for model input
            crop_eye = img_to_array(crop_eye)
            # Preprocess eye region
            crop_eye = preprocess_image(crop_eye)
            # Predict drowsiness using the trained model
            pred = drowse_model.predict(crop_eye)[0][0]
            # Put rectangle around the detected eyes and put the text based on the prediction
            if pred < 0.5:
                cv2.rectangle(img, (x,y), (x+l,y+w), (0,0,255), 2)
                cv2.putText(img, "Drowsy", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            else:
                cv2.rectangle(img, (x,y), (x+l,y+w), (0,255,0), 2)
                cv2.putText(img, "Not Drowsy", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        # Display the processed image with detections
        st.image(img, channels='BGR')

# Video input mode: Upload and process video to detect drowsiness frame by frame
elif choice == "VIDEO":
    # Provide an option to upload a video file for drowsiness detection
    file = st.file_uploader("Upload Video")
    # Initialize an empty Streamlit container for displaying video frames
    windows = st.empty()
    if file:
        # Save the uploaded video file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(file.read())
        # Load the video using OpenCV
        vid = cv2.VideoCapture(tmp_file.name)
        # Process the video frame-by-frame
        while(vid.isOpened()):
            flag, frame=vid.read()
            if (flag):
                # Detect eyes in the current frame
                eyes = eye_cascade.detectMultiScale(frame)
                # For each detected eye, process the eye region
                for (x,y,l,w) in eyes:
                    # Crop the eye region
                    crop_eye = frame[y:y+w, x:x+l]
                    # Convert the cropped eye image to an array
                    crop_eye = img_to_array(crop_eye)
                    # Preprocess the eye image for prediction
                    crop_eye = preprocess_image(crop_eye)
                    # Predict drowsiness using the loaded model
                    pred = drowse_model.predict(crop_eye)[0][0]
                    # Display results on the frame with rectangles on detected eyes
                    if pred < 0.5:
                        cv2.rectangle(frame, (x,y), (x+l,y+w), (0,0,255), 2)
                        cv2.putText(frame, "Drowsy", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    else:
                        cv2.rectangle(frame, (x,y), (x+l,y+w), (0,255, 0), 2)
                        cv2.putText(frame, "Not Drowsy", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                    # Display the processed frame in the Streamlit app
                    windows.image(frame, channels="BGR")

# Camera mode: Use webcam (or dashcam) or IP camera for live drowsiness detection
elif choice == "CAMERA":
    # Initialize camera session state and input field for camera source
    st.session_state["CAMERA"] = True
    k = st.text_input("Enter 0 to open webcam (or dashcam) or write URL for opening IP camera")
    # Convert single character input to integer for webcam access
    if len(k) == 1:
        k = int(k)
    # Buttons to start or stop the camera stream
    btn = st.button("Start Camera")
    cls_btn = st.button("Stop Camera")
    if cls_btn:
        st.session_state["CAMERA"] = False
    
    # Initialize an empty Streamlit container for displaying camera frames
    windows = st.empty()
    # If Start button is clicked and the camera session state is active
    if btn and st.session_state["CAMERA"]:
        # Capture video from the specified source (webcam/dashcam or IP camera)
        vid = cv2.VideoCapture(k)
        # Process the video frame-by-frame
        while(vid.isOpened()):
            flag, frame=vid.read()
            if (flag):
                # Detect eyes in the current frame
                eyes = eye_cascade.detectMultiScale(frame)
                # For each detected eye, process the eye region
                for (x,y,l,w) in eyes:
                    crop_eye = frame[y:y+w, x:x+l]
                    crop_eye = img_to_array(crop_eye)
                    crop_eye = preprocess_image(crop_eye)
                    pred = drowse_model.predict(crop_eye)[0][0]
                    # Display results on the frame with rectangles on detected eyes
                    if pred < 0.5:
                        cv2.rectangle(frame, (x,y), (x+l,y+w), (0,0,255), 2)
                        cv2.putText(frame, "Drowsy", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    else:
                        cv2.rectangle(frame, (x,y), (x+l,y+w), (0,255,0), 2)
                        cv2.putText(frame, "Not Drowsy", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                    # Display the processed frame in the Streamlit app
                    windows.image(frame, channels="BGR")
