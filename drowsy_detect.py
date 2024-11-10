# Import necessary libraries
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
import cv2

# Constants to control frame skipping and threshold for eye closure detection
SKIP_FRAMES_BY = 15
FRAMES_THRSH = 3
count = 0
eye_count = 0
eye_closed = 0

# Load the pre-trained drowsiness detection model
model_path = r"D:\ML_Projects\Drowsiness-Detection-System\Models\drowse_model_tf_7_epoch_250.h5"
model = load_model(model_path)
# Load Haar Cascade for eye detection
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Preprocess image to match the model's expected input format
def preprocess_image(img_array):
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    img_array = cv2.resize(img_array, (80, 80))
    img_array = img_array.astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Start capturing video from the webcam/dashcam or video file
vid = cv2.VideoCapture(0) #r"D:\ML_Projects\Drowsiness-Detection-System\Data\anp_test.mp4")
# Process video frames in a loop
while vid.isOpened():
    ret, frame = vid.read()
    if not ret:
        break
    count += 1
    score = 0
    # Process every 15th (defined by the constant SKIP_FRAMES_BY) frame to improve efficiency
    if (count % SKIP_FRAMES_BY) == 0:
        # Detect eyes in the current frame
        eyes = eye_cascade.detectMultiScale(frame, 1.1, 9)
        for(x, y, w, h) in eyes:
            eye_count += 1
            # Extract the eye region from the frame
            img = frame[y:y+h, x:x+w]
            #cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # Predict drowsiness based on the eye region
            pred = model.predict(preprocess_image(img))
            # Accumulate drowsiness score for each detected eye
            score += pred[0][0]
        # If no eyes are detected, skip to the next frame
        if eye_count == 0:
            continue
        else:
            # Calculate average score across detected eyes
            score = score / eye_count
            # Reset eye count for the next frame
            eye_count = 0
        # Check if the drowsiness score indicates eyes are closed
        if score < 0.7:
            if eye_closed < (FRAMES_THRSH+1):
                # Increment consecutive closed-eye counter
                eye_closed += 1
        else:
            # Reset counter if eyes are open
            eye_closed -= 1
            if eye_closed < 0:
                # Ensure counter doesn't go negative
                eye_closed = 0

    # Display "Drowsy" warning if eyes are closed for several consecutive frames
    if eye_closed > (FRAMES_THRSH-1):
        cv2.putText(frame, "Drowsy", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "Not Drowsy", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Show the processed frame with drowsiness status
    cv2.imshow("Drowsiness Detection", frame)
    
    # Exit if any key is pressed
    if cv2.waitKey(1) != -1:
        break
# Release video capture and close all OpenCV windows
vid.release()
cv2.destroyAllWindows()
