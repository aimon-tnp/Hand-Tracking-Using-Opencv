from pathlib import Path
import time

import cv2

from asl_wordle.landmarks import HandTracker, draw_landmarks, normalize_landmarks
from asl_wordle.recognizer import ASLLetterRecognizer, StableLetterInput
from asl_wordle.ui import WINDOW_NAME, render
from asl_wordle.wordle import WordleGame, load_words


ROOT = Path(__file__).resolve().parent
WORD_LIST_PATH = ROOT / "data" / "word_list.txt"


def build_game():
    words = load_words(WORD_LIST_PATH)
    return WordleGame(valid_words=set(words), answers=words)


def handle_key(key, game, stable_input):
    if key in (ord("q"), ord("Q")):
        return False
    if key in (8, 127, ord("d"), ord("D")):
        game.delete_letter()
        stable_input.reset()
    elif key in (10, 13):
        game.submit_guess()
        stable_input.reset()
    elif key in (ord("r"), ord("R")):
        game.restart()
        stable_input.reset()
    return True


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    game = build_game()
    tracker = HandTracker()
    recognizer = ASLLetterRecognizer()
    stable_input = StableLetterInput(hold_seconds=0.7)
    start_time = time.monotonic()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp_ms = int((time.monotonic() - start_time) * 1000)
            hand_landmarks = tracker.detect(frame_rgb, timestamp_ms)

            features = None
            if hand_landmarks:
                draw_landmarks(frame, hand_landmarks, tracker.connections)
                features = normalize_landmarks(hand_landmarks)

            prediction = recognizer.predict(features)
            accepted_letter, hold_progress = stable_input.update(prediction)
            if accepted_letter:
                game.add_letter(accepted_letter)

            canvas = render(frame, game, prediction, hold_progress, recognizer.is_ready)
            cv2.imshow(WINDOW_NAME, canvas)

            key = cv2.waitKey(1) & 0xFF
            if key != 255 and not handle_key(key, game, stable_input):
                break
    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
