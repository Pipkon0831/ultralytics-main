import cv2
import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO

model = YOLO('best.pt')

is_running = False

def detect_image():
    global is_running
    image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if image_path:
        img = cv2.imread(image_path)
        if img is not None:
            results = model.predict(source=img)
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls = int(box.cls)
                    confidence = box.conf[0]
                    class_name = model.names[cls]
                    x1, y1, x2, y2 = box.xyxy[0]
                    color = get_color_for_class(cls)
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.putText(img, f"{class_name}: {confidence:.2f}", (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            image_window = tk.Toplevel(root)
            image_window.title("Image Detection")
            from PIL import Image, ImageTk
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            img_tk = ImageTk.PhotoImage(img_pil)
            label = tk.Label(image_window, image=img_tk)
            label.image = img_tk
            label.pack()
        else:
            print(f"无法读取图片：{image_path}")

def detect_video():
    global is_running
    video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mkv")])
    if video_path:
        cap = cv2.VideoCapture(video_path)
        is_running = True
        while cap.isOpened() and is_running:
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(source=frame)
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls = int(box.cls)
                    confidence = box.conf[0]
                    class_name = model.names[cls]
                    x1, y1, x2, y2 = box.xyxy[0]
                    color = get_color_for_class(cls)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.putText(frame, f"{class_name}: {confidence:.2f}", (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # 缩放视频帧
            scale_percent = 50  # 设置缩放比例为50%
            width = int(frame.shape[1] * scale_percent / 100)
            height = int(frame.shape[0] * scale_percent / 100)
            dim = (width, height)

            # 调整图像大小
            resized_frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

            cv2.imshow('Video Detection', resized_frame)
            key = cv2.waitKey(1)
            if key == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

def start_camera_detection():
    global is_running
    cap = cv2.VideoCapture(0)
    is_running = True
    while cap.isOpened() and is_running:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls)
                confidence = box.conf[0]
                class_name = model.names[cls]
                x1, y1, x2, y2 = box.xyxy[0]
                color = get_color_for_class(cls)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, f"{class_name}: {confidence:.2f}", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow('Camera Detection', frame)
        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

def on_closing():
    global is_running
    is_running = False
    cv2.destroyAllWindows()

def get_color_for_class(cls):
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
    return colors[cls % len(colors)]

root = tk.Tk()
root.title('检测界面')

image_button = tk.Button(root, text="图片检测", command=detect_image, bg="#4CAF50", fg="white", font=("Helvetica", 12))
video_button = tk.Button(root, text="视频检测", command=detect_video, bg="#2196F3", fg="white", font=("Helvetica", 12))
camera_button = tk.Button(root, text="摄像头检测", command=start_camera_detection, bg="#FF9800", fg="white", font=("Helvetica", 12))

image_button.pack(pady=20)
video_button.pack(pady=20)
camera_button.pack(pady=20)

root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()
