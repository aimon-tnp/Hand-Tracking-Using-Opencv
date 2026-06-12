from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


HAND_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "hand_landmarker.task"


def normalize_landmarks(hand_landmarks):
    points = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=np.float32)
    wrist = points[0].copy()
    points -= wrist

    span = np.max(np.ptp(points[:, :2], axis=0))
    if span < 1e-6:
        span = 1.0
    points /= span
    return points.flatten()


def draw_landmarks(image, hand_landmarks, connections):
    image_height, image_width, _ = image.shape

    for connection in connections:
        start = hand_landmarks[connection.start]
        end = hand_landmarks[connection.end]
        start_point = (int(start.x * image_width), int(start.y * image_height))
        end_point = (int(end.x * image_width), int(end.y * image_height))
        cv2.line(image, start_point, end_point, (224, 224, 224), 2)

    for index, landmark in enumerate(hand_landmarks):
        center = (int(landmark.x * image_width), int(landmark.y * image_height))
        radius = 6 if index == 0 else 3
        color = (255, 0, 255) if index == 0 else (0, 0, 255)
        cv2.circle(image, center, radius, color, cv2.FILLED)


class HandTracker:
    def __init__(self, model_path=HAND_MODEL_PATH, running_mode=None):
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Missing MediaPipe hand model: {model_path}\n"
                "Download hand_landmarker.task into the models directory."
            )

        from mediapipe.tasks.python.vision import hand_landmarker

        self.connections = hand_landmarker.HandLandmarksConnections.HAND_CONNECTIONS
        self.running_mode = running_mode or mp.tasks.vision.RunningMode.VIDEO
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=self.running_mode,
            num_hands=1,
        )
        self.landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

    def detect(self, image_rgb, timestamp_ms=0):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        if self.running_mode == mp.tasks.vision.RunningMode.IMAGE:
            result = self.landmarker.detect(mp_image)
        else:
            result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        return result.hand_landmarks[0] if result.hand_landmarks else None

    def close(self):
        self.landmarker.close()
