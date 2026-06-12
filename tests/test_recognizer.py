import unittest

from asl_wordle.recognizer import Prediction, StableLetterInput


class StableLetterInputTest(unittest.TestCase):
    def test_accepts_letter_after_hold_time(self):
        stable_input = StableLetterInput(hold_seconds=0.7)
        prediction = Prediction("A", 0.9, "Hold steady.")

        self.assertEqual(stable_input.update(prediction, now=1.0), (None, 0.0))
        self.assertEqual(stable_input.update(prediction, now=1.5), (None, 0.7142857142857143))
        self.assertEqual(stable_input.update(prediction, now=1.7), ("A", 1.0))
        self.assertEqual(stable_input.update(prediction, now=2.0), (None, 1.0))

    def test_resets_when_prediction_disappears(self):
        stable_input = StableLetterInput(hold_seconds=0.7)
        stable_input.update(Prediction("A", 0.9, "Hold steady."), now=1.0)

        self.assertEqual(stable_input.update(Prediction(None, 0.0, "No hand."), now=1.2), (None, 0.0))
        self.assertIsNone(stable_input.current_letter)


if __name__ == "__main__":
    unittest.main()
