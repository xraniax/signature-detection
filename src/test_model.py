from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image

def test_model(img_path):
    model = YOLO("runs/detect/signature_detector4/weights/best.pt")

    results = model(img_path)

    img = Image.open(img_path)
    plt.imshow(img)
    plt.axis("off")

    for b in results[0].boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = b
        plt.gca().add_patch(
            plt.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                edgecolor='red',
                linewidth=2
            )
        )

    plt.show()

if __name__ == "__main__":
    path = input("Enter image path: ")
    test_model(path)
