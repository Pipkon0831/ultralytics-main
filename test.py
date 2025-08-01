import cv2
import os
import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO

# 加载模型
model = YOLO('best.pt')

def open_file():
    file_path = filedialog.askopenfilename()
    if os.path.isfile(file_path):
        if file_path.endswith('.jpg') or file_path.endswith('.png'):
            image = cv2.imread(file_path)
            results = model.predict(source=image)
            result = results[0]
            annotated_image = result.plot()
            cv2.imshow('Image Detection', annotated_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        elif file_path.endswith('.mp4'):
            cap = cv2.VideoCapture(file_path)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                results = model.predict(source=frame)
                result = results[0]
                annotated_frame = result.plot()
                cv2.imshow('Video Detection', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()

def start_camera_detection():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(source=frame)
        result = results[0]
        annotated_frame = result.plot()
        cv2.imshow('Real-time Monitoring', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

root = tk.Tk()
root.title("Object Detection")

open_button = tk.Button(root, text="Open File", command=open_file)
open_button.pack()

camera_button = tk.Button(root, text="Start Camera Detection", command=start_camera_detection)
camera_button.pack()

root.mainloop()