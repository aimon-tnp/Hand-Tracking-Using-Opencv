from pathlib import Path
import time

import cv2
import mediapipe as mp


MODEL_PATH = Path(__file__).with_name("models") / "hand_landmarker.task"
VIDEO_PATH = "a.mp4"
OUTPUT_VIDEO_PATH = "output_video.mp4"


def draw_task_landmarks(image, hand_landmarks, connections):
    image_height, image_width, _ = image.shape

    for landmark_id, landmark in enumerate(hand_landmarks):
        cx = int(landmark.x * image_width)
        cy = int(landmark.y * image_height)
        print(landmark_id, cx, cy)
        if landmark_id == 0:
            cv2.circle(image, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

    for connection in connections:
        start = hand_landmarks[connection.start]
        end = hand_landmarks[connection.end]
        start_point = (int(start.x * image_width), int(start.y * image_height))
        end_point = (int(end.x * image_width), int(end.y * image_height))
        cv2.line(image, start_point, end_point, (224, 224, 224), 2)

    for landmark in hand_landmarks:
        center = (int(landmark.x * image_width), int(landmark.y * image_height))
        cv2.circle(image, center, 3, (0, 0, 255), cv2.FILLED)


def write_with_legacy_solutions(cap, out):
    mphands = mp.solutions.hands
    hands = mphands.Hands(False)
    mp_draw = mp.solutions.drawing_utils
    previous_time = 0

    while True:
        success, image = cap.read()
        if not success:
            break

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for landmark_id, landmark in enumerate(hand_landmarks.landmark):
                    height, width, _ = image.shape
                    cx = int(landmark.x * width)
                    cy = int(landmark.y * height)
                    print(landmark_id, cx, cy)
                    if landmark_id == 0:
                        cv2.circle(image, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

                mp_draw.draw_landmarks(image, hand_landmarks, mphands.HAND_CONNECTIONS)

        current_time = time.time()
        fps = 1 / (current_time - previous_time) if previous_time else 0
        previous_time = current_time

        cv2.putText(image, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow("Image", image)
        out.write(image)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    hands.close()


def write_with_tasks_api(cap, out, input_fps):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing MediaPipe model: {MODEL_PATH}\n"
            "Download it from:\n"
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/latest/hand_landmarker.task"
        )

    from mediapipe.tasks.python.vision import hand_landmarker

    base_options = mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
    )
    connections = hand_landmarker.HandLandmarksConnections.HAND_CONNECTIONS
    previous_time = 0
    frame_index = 0
    frame_duration_ms = 1000 / input_fps if input_fps > 0 else 33.33

    with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            success, image = cap.read()
            if not success:
                break

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            timestamp_ms = int(frame_index * frame_duration_ms)
            frame_index += 1
            results = landmarker.detect_for_video(mp_image, timestamp_ms)

            for hand_landmarks in results.hand_landmarks:
                draw_task_landmarks(image, hand_landmarks, connections)

            current_time = time.time()
            fps = 1 / (current_time - previous_time) if previous_time else 0
            previous_time = current_time

            cv2.putText(image, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
            cv2.imshow("Image", image)
            out.write(image)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {VIDEO_PATH}")

    input_fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, input_fps, (width, height))

    try:
        if hasattr(mp, "solutions"):
            write_with_legacy_solutions(cap, out)
        else:
            write_with_tasks_api(cap, out, input_fps)
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
