# ASL Wordle Using OpenCV and MediaPipe

This project is an OpenCV webcam version of Wordle that accepts guesses through fingerspelled ASL letters. MediaPipe extracts hand landmarks from the camera feed, a trained classifier predicts the ASL alphabet letter, and the app fills a 5-letter Wordle board after the predicted letter is held steady.

## Features

- Mirrored webcam feed with MediaPipe hand landmarks.
- OpenCV-drawn Wordle board, prediction status, confidence, and controls.
- Stable-hold letter entry to reduce accidental inputs.
- Keyboard controls for submit, delete, restart, and quit.
- Training script for an ASL alphabet classifier using hand landmark features.

## Setup

```bash
pip install -r requirements.txt
```

The app expects the MediaPipe hand model at:

```text
models/hand_landmarker.task
```

If it is missing, download it from:

```text
https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

## Train the ASL Classifier

Download and extract the Kaggle ASL Alphabet dataset:

```text
https://www.kaggle.com/datasets/grassknoted/asl-alphabet
```

Then train:

```bash
python scripts/train_asl_classifier.py /path/to/asl_alphabet_dataset
```

This writes:

```text
models/asl_letter_classifier.pkl
```

For a faster first pass, limit samples per class:

```bash
python scripts/train_asl_classifier.py /path/to/asl_alphabet_dataset --max-per-class 150
```

The v1 game excludes `J` and `Z` from the answer list because those letters are dynamic in ASL.

## Run

```bash
python app.py
```

Controls:

- Hold a recognized ASL letter for about 0.7 seconds to enter it.
- `Enter`: submit a 5-letter guess.
- `Backspace` or `D`: delete the last letter.
- `R`: restart with a new word.
- `Q`: quit.

If `models/asl_letter_classifier.pkl` does not exist yet, the app still opens, but it will show a message telling you to train the classifier first.

## Tests

```bash
python -m unittest discover -s tests
```
