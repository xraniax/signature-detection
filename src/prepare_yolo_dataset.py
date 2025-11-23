import os
import shutil
from datasets import load_dataset
from PIL import Image

def save_yolo_dataset(data, save_dir="yolo_data", subset_size=300):

    train_dir = f"{save_dir}/images/train"
    label_dir = f"{save_dir}/labels/train"

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    dataset_subset = data['train'].select(range(subset_size))

    for i, sample in enumerate(dataset_subset):
        img = sample['document']
        bboxes = sample['bbox']

        img_path = f"{train_dir}/doc_{i}.png"
        img.save(img_path)

        w, h = img.size
        label_path = f"{label_dir}/doc_{i}.txt"

        with open(label_path, "w") as f:
            for box in bboxes:
                
                x_min, y_min, x_max, y_max = box

                # Bounding boxes are already normalized, no need to divide by w/h
                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2
                width = x_max - x_min
                height = y_max - y_min

                f.write(f"0 {x_center} {y_center} {width} {height}\n")

    print(f"Saved {subset_size} samples to {save_dir}")


def create_validation_split(save_dir="yolo_data", val_size=30):
    print("Creating validation split...")

    os.makedirs(f"{save_dir}/images/val", exist_ok=True)
    os.makedirs(f"{save_dir}/labels/val", exist_ok=True)

    for i in range(val_size):
        shutil.move(f"{save_dir}/images/train/doc_{i}.png", f"{save_dir}/images/val/")
        shutil.move(f"{save_dir}/labels/train/doc_{i}.txt", f"{save_dir}/labels/val/")

    print("Validation set created.")


if __name__ == "__main__":
    data = load_dataset("Mels22/SigDetectVerifyFlow")
    save_yolo_dataset(data)
    create_validation_split()
