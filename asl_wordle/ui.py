import cv2
import numpy as np

from asl_wordle.wordle import MAX_ATTEMPTS, WORD_LENGTH, TileState


WINDOW_NAME = "ASL Wordle"
CANVAS_WIDTH = 1180
CANVAS_HEIGHT = 720
CAMERA_WIDTH = 760
SIDEBAR_X = 790

COLORS = {
    TileState.EMPTY: (58, 58, 60),
    TileState.ABSENT: (58, 58, 60),
    TileState.PRESENT: (0, 180, 200),
    TileState.CORRECT: (83, 141, 78),
}


def draw_text(image, text, origin, scale=0.7, color=(245, 245, 245), thickness=2):
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def fit_camera_frame(frame):
    height, width, _ = frame.shape
    scale = min(CAMERA_WIDTH / width, CANVAS_HEIGHT / height)
    resized = cv2.resize(frame, (int(width * scale), int(height * scale)))
    output = np.zeros((CANVAS_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
    y = (CANVAS_HEIGHT - resized.shape[0]) // 2
    x = (CAMERA_WIDTH - resized.shape[1]) // 2
    output[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return output


def draw_wordle_board(canvas, game):
    tile_size = 58
    gap = 8
    start_x = SIDEBAR_X
    start_y = 92

    for row in range(MAX_ATTEMPTS):
        if row < len(game.guesses):
            letters = list(game.guesses[row].word)
            states = game.guesses[row].states
        elif row == len(game.guesses):
            letters = game.current_guess + [""] * (WORD_LENGTH - len(game.current_guess))
            states = [TileState.EMPTY] * WORD_LENGTH
        else:
            letters = [""] * WORD_LENGTH
            states = [TileState.EMPTY] * WORD_LENGTH

        for col in range(WORD_LENGTH):
            x = start_x + col * (tile_size + gap)
            y = start_y + row * (tile_size + gap)
            color = COLORS[states[col]]
            cv2.rectangle(canvas, (x, y), (x + tile_size, y + tile_size), color, cv2.FILLED)
            cv2.rectangle(canvas, (x, y), (x + tile_size, y + tile_size), (120, 120, 120), 2)
            if letters[col]:
                draw_text(canvas, letters[col], (x + 18, y + 40), 1.0, (255, 255, 255), 2)


def render(frame, game, prediction, hold_progress, classifier_ready):
    canvas = np.full((CANVAS_HEIGHT, CANVAS_WIDTH, 3), 24, dtype=np.uint8)
    canvas[:, :CAMERA_WIDTH] = fit_camera_frame(frame)
    cv2.line(canvas, (CAMERA_WIDTH + 14, 0), (CAMERA_WIDTH + 14, CANVAS_HEIGHT), (70, 70, 70), 2)

    draw_text(canvas, "ASL WORDLE", (SIDEBAR_X, 42), 1.1, (255, 255, 255), 3)
    draw_wordle_board(canvas, game)

    letter = prediction.letter or "-"
    confidence = int(prediction.confidence * 100)
    draw_text(canvas, f"Letter: {letter}", (SIDEBAR_X, 530), 0.8)
    draw_text(canvas, f"Confidence: {confidence}%", (SIDEBAR_X, 565), 0.7)

    bar_x, bar_y, bar_w, bar_h = SIDEBAR_X, 585, 300, 14
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), cv2.FILLED)
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + int(bar_w * hold_progress), bar_y + bar_h), (255, 0, 255), cv2.FILLED)

    status_color = (180, 180, 180) if classifier_ready else (80, 190, 255)
    draw_text(canvas, prediction.status, (SIDEBAR_X, 628), 0.55, status_color, 1)
    draw_text(canvas, game.message, (SIDEBAR_X, 656), 0.55, (230, 230, 230), 1)
    draw_text(canvas, "Enter submit | Backspace/D delete | R restart | Q quit", (SIDEBAR_X, 690), 0.45, (180, 180, 180), 1)
    return canvas
