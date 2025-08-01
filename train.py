from ultralytics import YOLO

model = YOLO('yolov8n.pt')

model.train(data='safehat.yaml',epochs=50)

model.val()