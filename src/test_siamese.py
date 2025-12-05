import torch
import cv2
import numpy as np
from siamese_resnet import SiameseResNet
from torchvision import transforms

MODEL_PATH = "../models/final_model.pth"

# ----------------------------
# Load model
# ----------------------------
def load_model(model_path):
    checkpoint = torch.load(model_path, map_location="cpu")
    
    model = SiameseResNet(
        embedding_dim=checkpoint["config"]["embedding_dim"],
        pretrained=False
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint["config"]


# ----------------------------
# Image preprocessing
# ----------------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])


def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Image not found: {path}")
    return transform(img).unsqueeze(0)  # (1, 3, 224, 224)


# ----------------------------
# Predict similarity
# ----------------------------
def predict(model, config, img1_path, img2_path):
    img1 = load_image(img1_path)
    img2 = load_image(img2_path)

    with torch.no_grad():
        emb1, emb2 = model(img1, img2)
        distance = torch.nn.functional.pairwise_distance(emb1, emb2).item()

    threshold = config["threshold"]
    is_genuine = distance < threshold

    return distance, is_genuine


# ----------------------------
# MAIN TEST
# ----------------------------
if __name__ == "__main__":
    print("Loading model...")
    model, config = load_model(MODEL_PATH)

    print("\nEnter the path for the FIRST signature image:")
    img1 = input("> ")

    print("Enter the path for the SECOND signature image:")
    img2 = input("> ")

    print("\nTesting...")
    distance, is_genuine = predict(model, config, img1, img2)

    print("\n----------------------------")
    print(f"Distance: {distance:.4f}")
    print("Result:", "Genuine (same person)" if is_genuine else "Forged (different person)")
    print("----------------------------")
