from ultralytics import YOLO

def train_model():
    print("Starting YOLOv8 training...")
    model = YOLO("yolov8s.pt")  # higher accuracy than yolov8n.pt

    model.train(
        data="configs/signature.yaml",
        epochs=50,       # stronger training
        imgsz=640,
        batch=8,
        name="signature_detector",
        workers=4
    )

    print("Training completed.")

if __name__ == "__main__":
    train_model()
