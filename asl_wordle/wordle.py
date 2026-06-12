from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
import random


WORD_LENGTH = 5
MAX_ATTEMPTS = 6


class TileState(str, Enum):
    EMPTY = "empty"
    ABSENT = "absent"
    PRESENT = "present"
    CORRECT = "correct"


@dataclass
class GuessResult:
    word: str
    states: list[TileState]


@dataclass
class WordleGame:
    valid_words: set[str]
    answers: list[str]
    answer: str | None = None
    current_guess: list[str] = field(default_factory=list)
    guesses: list[GuessResult] = field(default_factory=list)
    message: str = "Hold an ASL letter to enter it."
    won: bool = False
    lost: bool = False

    def __post_init__(self):
        if not self.answers:
            raise ValueError("answers must contain at least one word")
        if self.answer is None:
            self.answer = random.choice(self.answers)
        self.answer = self.answer.upper()
        self.valid_words = {word.upper() for word in self.valid_words}

    @property
    def is_over(self):
        return self.won or self.lost

    def add_letter(self, letter):
        if self.is_over:
            self.message = "Game over. Press R for a new word."
            return False
        if len(self.current_guess) >= WORD_LENGTH:
            self.message = "Press Enter to submit."
            return False

        self.current_guess.append(letter.upper())
        self.message = "".join(self.current_guess)
        return True

    def delete_letter(self):
        if self.is_over:
            self.message = "Game over. Press R for a new word."
            return False
        if not self.current_guess:
            self.message = "Nothing to delete."
            return False

        self.current_guess.pop()
        self.message = "".join(self.current_guess) or "Hold an ASL letter to enter it."
        return True

    def submit_guess(self):
        if self.is_over:
            self.message = "Game over. Press R for a new word."
            return False
        if len(self.current_guess) != WORD_LENGTH:
            self.message = "Guess must be 5 letters."
            return False

        word = "".join(self.current_guess).upper()
        if word not in self.valid_words:
            self.message = f"{word} is not in the word list."
            return False

        states = score_guess(word, self.answer)
        self.guesses.append(GuessResult(word, states))
        self.current_guess.clear()

        if word == self.answer:
            self.won = True
            self.message = f"You got it: {self.answer}"
        elif len(self.guesses) >= MAX_ATTEMPTS:
            self.lost = True
            self.message = f"Answer: {self.answer}"
        else:
            self.message = "Next guess."
        return True

    def restart(self, answer=None):
        self.answer = (answer or random.choice(self.answers)).upper()
        self.current_guess.clear()
        self.guesses.clear()
        self.message = "Hold an ASL letter to enter it."
        self.won = False
        self.lost = False


def score_guess(guess, answer):
    guess = guess.upper()
    answer = answer.upper()
    states = [TileState.ABSENT] * len(guess)
    remaining = Counter()

    for index, letter in enumerate(guess):
        if letter == answer[index]:
            states[index] = TileState.CORRECT
        else:
            remaining[answer[index]] += 1

    for index, letter in enumerate(guess):
        if states[index] == TileState.CORRECT:
            continue
        if remaining[letter] > 0:
            states[index] = TileState.PRESENT
            remaining[letter] -= 1

    return states


def load_words(path, unsupported_letters=frozenset({"J", "Z"})):
    words = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            word = line.strip().upper()
            if len(word) != WORD_LENGTH or not word.isalpha():
                continue
            if any(letter in unsupported_letters for letter in word):
                continue
            words.append(word)
    return words
