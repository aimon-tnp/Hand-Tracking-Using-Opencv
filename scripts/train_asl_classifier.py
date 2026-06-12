import argparse
from collections import Counter
from pathlib import Path
import pickle
import sys

import cv2
import mediapipe as mp
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from asl_wordle.landmarks import HAND_MODEL_PATH, HandTracker, normalize_landmarks


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EXCLUDED_LABELS = {"DELETE", "DEL", "NOTHING", "SPACE", "J", "Z"}


def iter_label_dirs(dataset_dir):
    candidates = [
        dataset_dir / "asl_alphabet_train" / "asl_alphabet_train",
        dataset_dir / "asl_alphabet_train",
        dataset_dir,
    ]

    for root in candidates:
        if not root.exists():
            continue
        label_dirs = [path for path in root.iterdir() if path.is_dir()]
        label_names = {path.name.upper() for path in label_dirs}
        if len(label_names & set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")) >= 20:
            return sorted(label_dirs)
    return []


def iter_images(label_dir, max_per_class=None):
    count = 0
    for path in sorted(label_dir.rglob("*")):
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        yield path
        count += 1
        if max_per_class and count >= max_per_class:
            break


def extract_dataset(dataset_dir, max_per_class):
    tracker = HandTracker(model_path=HAND_MODEL_PATH, running_mode=mp.tasks.vision.RunningMode.IMAGE)
    features = []
    labels = []
    skipped = Counter()

    try:
        for label_dir in iter_label_dirs(dataset_dir):
            label = label_dir.name.upper()
            if label in EXCLUDED_LABELS or len(label) != 1:
                continue

            for image_path in iter_images(label_dir, max_per_class=max_per_class):
                image = cv2.imread(str(image_path))
                if image is None:
                    skipped[label] += 1
                    continue

                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                hand_landmarks = tracker.detect(image_rgb)
                if hand_landmarks is None:
                    skipped[label] += 1
                    continue

                features.append(normalize_landmarks(hand_landmarks))
                labels.append(label)
    finally:
        tracker.close()

    return np.asarray(features, dtype=np.float32), np.asarray(labels), skipped


def train_classifier(features, labels):
    model = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=5, weights="distance"),
    )
    model.fit(features, labels)
    return model


def main():
    parser = argparse.ArgumentParser(description="Train the ASL Wordle letter classifier.")
    parser.add_argument("dataset_dir", type=Path, help="Path to the extracted ASL alphabet dataset.")
    parser.add_argument("--output", type=Path, default=Path("models/asl_letter_classifier.pkl"))
    parser.add_argument("--max-per-class", type=int, default=600)
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    features, labels, skipped = extract_dataset(args.dataset_dir, args.max_per_class)
    if len(labels) == 0:
        raise RuntimeError("No usable hand landmark samples were extracted.")

    class_counts = Counter(labels)
    print("Extracted samples by class:")
    for label, count in sorted(class_counts.items()):
        print(f"  {label}: {count}")

    if skipped:
        print("Skipped images with no usable hand landmarks:")
        for label, count in sorted(skipped.items()):
            print(f"  {label}: {count}")

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=args.test_size,
        random_state=42,
        stratify=labels,
    )
    model = train_classifier(x_train, y_train)
    predictions = model.predict(x_test)

    print("\nClassification report:")
    print(classification_report(y_test, predictions))
    print("Confusion matrix labels:", " ".join(model.classes_))
    print(confusion_matrix(y_test, predictions, labels=model.classes_))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as file:
        pickle.dump({"model": model, "classes": model.classes_.tolist()}, file)
    print(f"\nSaved classifier to {args.output}")


if __name__ == "__main__":
    main()
