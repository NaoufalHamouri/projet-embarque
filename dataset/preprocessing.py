import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

# Paths
DATA_DIR = Path("data")
OUTPUT_DIR = Path("processed")

def collect_images():
    images, labels = [], []
    for split in ["train", "val", "test"]:
        for label, cls in enumerate(["NORMAL", "PNEUMONIA"]):
            folder = DATA_DIR / split / cls
            for img in folder.glob("*.jpeg"):
                images.append(str(img))
                labels.append(label)
    return images, labels

def create_split(images, labels, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    for cls in ["NORMAL", "PNEUMONIA"]:
        (output_dir / cls).mkdir(exist_ok=True)
    for img_path, label in zip(images, labels):
        cls = "NORMAL" if label == 0 else "PNEUMONIA"
        dst = output_dir / cls / Path(img_path).name
        shutil.copy2(img_path, dst)

def main():
    print("Collecting all images...")
    images, labels = collect_images()
    print(f"Total images: {len(images)}")

    # Split: 70% train, 15% val, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        images, labels, test_size=0.30, random_state=42, stratify=labels)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    print("Creating processed splits...")
    create_split(X_train, y_train, OUTPUT_DIR / "train")
    create_split(X_val, y_val, OUTPUT_DIR / "val")
    create_split(X_test, y_test, OUTPUT_DIR / "test")

    print("Done! Dataset ready at dataset/processed/")

    # Print class distribution
    for split in ["train", "val", "test"]:
        n = len(list((OUTPUT_DIR / split / "NORMAL").glob("*")))
        p = len(list((OUTPUT_DIR / split / "PNEUMONIA").glob("*")))
        print(f"{split}: NORMAL={n}, PNEUMONIA={p}")

if __name__ == "__main__":
    main()
