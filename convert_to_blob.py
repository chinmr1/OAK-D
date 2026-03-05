from ultralytics import YOLO

model = YOLO(r"runs\detect\train\weights\best.pt")
model.export(format="onnx", imgsz=640, opset=12)

import blobconverter

blob_path = blobconverter.from_onnx(
    model="runs/detect/train/weights/best.onnx",
    shaves=6,
    version="2022.1",
)

print(blob_path)
print(f"Blob saved to: {blob_path}")