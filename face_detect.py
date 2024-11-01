from ultralytics import YOLO
import cv2

model = YOLO(r"D:\ML_Projects\Drowsiness-Detection-System\Models\yolov8n-face.pt")
vid_path = r"D:\ML_Projects\Drowsiness-Detection-System\Data\group_2.mp4"

vid = cv2.VideoCapture(vid_path)
while vid.isOpened():
    ret, frame = vid.read()

    if not ret:
        break
    results = model(frame)
    boxes = results[0].boxes

    for box in boxes:
        top_left_x = int(box.xyxy.tolist()[0][0])
        top_left_y = int(box.xyxy.tolist()[0][1])
        bottom_right_x = int(box.xyxy.tolist()[0][2])
        bottom_right_y = int(box.xyxy.tolist()[0][3])

        cv2.rectangle(frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (50, 200, 129), 3)

    cv2.imshow("test", frame)
    cv2.waitKey(1)

vid.release()
cv2.destroyAllWindows()