import unittest

from asl_wordle.wordle import TileState, WordleGame, score_guess


class ScoreGuessTest(unittest.TestCase):
    def test_all_correct(self):
        self.assertEqual(
            score_guess("APPLE", "APPLE"),
            [TileState.CORRECT] * 5,
        )

    def test_repeated_letters_do_not_overcount_yellow(self):
        self.assertEqual(
            score_guess("ALLEY", "BASIC"),
            [
                TileState.PRESENT,
                TileState.ABSENT,
                TileState.ABSENT,
                TileState.ABSENT,
                TileState.ABSENT,
            ],
        )

    def test_correct_letters_are_counted_before_present_letters(self):
        self.assertEqual(
            score_guess("SHEEP", "STEEL"),
            [
                TileState.CORRECT,
                TileState.ABSENT,
                TileState.CORRECT,
                TileState.CORRECT,
                TileState.ABSENT,
            ],
        )


class WordleGameTest(unittest.TestCase):
    def test_add_delete_and_submit_win(self):
        game = WordleGame(valid_words={"APPLE"}, answers=["APPLE"], answer="APPLE")
        for letter in "APPLE":
            self.assertTrue(game.add_letter(letter))

        self.assertEqual(game.current_guess, list("APPLE"))
        self.assertTrue(game.submit_guess())
        self.assertTrue(game.won)
        self.assertEqual(game.guesses[0].word, "APPLE")

    def test_rejects_invalid_word(self):
        game = WordleGame(valid_words={"APPLE"}, answers=["APPLE"], answer="APPLE")
        for letter in "ABOUT":
            game.add_letter(letter)

        self.assertFalse(game.submit_guess())
        self.assertFalse(game.guesses)
        self.assertEqual(game.current_guess, list("ABOUT"))

    def test_delete_letter(self):
        game = WordleGame(valid_words={"APPLE"}, answers=["APPLE"], answer="APPLE")
        game.add_letter("A")
        self.assertTrue(game.delete_letter())
        self.assertEqual(game.current_guess, [])


if __name__ == "__main__":
    unittest.main()
