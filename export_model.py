# Run ONCE: downloads YOLOv8n (small, fast) and converts it to ONNX so that
# OpenCV's dnn module can load it. Creates yolov8n.onnx in this folder.

from ultralytics import YOLO

model = YOLO("yolov8s.pt")
model.export(format="onnx", imgsz=960, opset=12)
 