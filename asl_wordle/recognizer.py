from dataclasses import dataclass
from pathlib import Path
import pickle
import time

import numpy as np


CLASSIFIER_PATH = Path(__file__).resolve().parents[1] / "models" / "asl_letter_classifier.pkl"


@dataclass
class Prediction:
    letter: str | None
    confidence: float
    status: str


class ASLLetterRecognizer:
    def __init__(self, model_path=CLASSIFIER_PATH, min_confidence=0.65):
        self.model_path = Path(model_path)
        self.min_confidence = min_confidence
        self.model = None
        self.classes = None

        if self.model_path.exists():
            with open(self.model_path, "rb") as file:
                payload = pickle.load(file)
            self.model = payload["model"]
            self.classes = np.array(payload["classes"])

    @property
    def is_ready(self):
        return self.model is not None

    def predict(self, features):
        if not self.is_ready:
            return Prediction(None, 0.0, "Train models/asl_letter_classifier.pkl first.")
        if features is None:
            return Prediction(None, 0.0, "No hand detected.")

        sample = np.asarray(features, dtype=np.float32).reshape(1, -1)
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(sample)[0]
            best_index = int(np.argmax(probabilities))
            confidence = float(probabilities[best_index])
            letter = str(self.classes[best_index])
        else:
            letter = str(self.model.predict(sample)[0])
            confidence = 1.0

        if confidence < self.min_confidence:
            return Prediction(None, confidence, f"Low confidence: {letter}")
        return Prediction(letter, confidence, "Hold steady.")


class StableLetterInput:
    def __init__(self, hold_seconds=0.7):
        self.hold_seconds = hold_seconds
        self.current_letter = None
        self.started_at = None
        self.accepted_current = False

    def update(self, prediction, now=None):
        now = now or time.monotonic()
        if prediction.letter is None:
            self.reset()
            return None, 0.0

        if prediction.letter != self.current_letter:
            self.current_letter = prediction.letter
            self.started_at = now
            self.accepted_current = False
            return None, 0.0

        elapsed = now - self.started_at
        progress = min(elapsed / self.hold_seconds, 1.0)
        if elapsed >= self.hold_seconds and not self.accepted_current:
            self.accepted_current = True
            return self.current_letter, progress
        return None, progress

    def reset(self):
        self.current_letter = None
        self.started_at = None
        self.accepted_current = False
