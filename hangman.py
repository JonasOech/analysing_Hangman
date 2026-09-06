"""Hangman framework over the German noun list.

Typical use from a notebook:

    import hangman
    words = hangman.load_words()                 # dict[int, tuple[str, ...]]
    hangman.evaluate(my_strategy, words, n_games=50)   # 50 games per word length

A strategy is a function ``GameState -> str`` returning a single letter it has
not guessed before.  It never sees the secret word.
"""

from __future__ import annotations

import random
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Iterable, Sequence

CSV_PATH = "nouns.csv"

# 26 latin letters plus the German extras kept by the wordlist filter.
ALPHABET: tuple[str, ...] = tuple("abcdefghijklmnopqrstuvwxyzäöüß")

# Folding maps each umlaut onto its base letter; used only when a game is
# created with fold_umlauts=True, which shrinks the alphabet to 26.
_FOLD = {"ä": "a", "ö": "o", "ü": "u", "ß": "s"}
FOLDED_ALPHABET: tuple[str, ...] = tuple("abcdefghijklmnopqrstuvwxyz")

_WORD_RE = re.compile(r"[A-Za-zÄÖÜäöüß]+")

MASK = "_"


# --------------------------------------------------------------------------- #
# Word list
# --------------------------------------------------------------------------- #

def fold(word: str) -> str:
    """Map umlauts/ß onto their base letters (ä->a, ö->o, ü->u, ß->s)."""
    return "".join(_FOLD.get(c, c) for c in word)


@lru_cache(maxsize=None)
def load_words(csv_path: str = CSV_PATH, fold_umlauts: bool = False) -> dict[int, tuple[str, ...]]:
    """Load the noun lemmas, bucketed by length.

    Returns ``{length: (word, ...)}`` with lowercased, deduplicated plain
    strings.  Cached, so repeated calls are free.  ``.lower()`` is used rather
    than ``.casefold()`` on purpose: casefold turns "ß" into "ss" and would
    change word lengths.
    """
    import pandas as pd

    lemma = pd.read_csv(csv_path, low_memory=False, usecols=["lemma"])["lemma"].astype(str)
    lemma = lemma[lemma.str.fullmatch(_WORD_RE.pattern)].str.lower()
    if fold_umlauts:
        lemma = lemma.map(fold)
    words = sorted(set(lemma))

    buckets: dict[int, list[str]] = {}
    for w in words:
        buckets.setdefault(len(w), []).append(w)
    return {n: tuple(ws) for n, ws in sorted(buckets.items())}


def letter_frequency(words: Iterable[str], by_word: bool = True) -> dict[str, float]:
    """Relative letter frequency over ``words``.

    ``by_word=True`` counts each letter once per word it appears in, which is
    what matters in hangman (a guess reveals every copy at once).
    """
    counts: Counter[str] = Counter()
    total = 0
    for w in words:
        for c in (set(w) if by_word else w):
            counts[c] += 1
            total += 1
    return {c: n / total for c, n in counts.most_common()} if total else {}


# --------------------------------------------------------------------------- #
# Game state handed to a strategy
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class GameState:
    """Everything a strategy is allowed to know. The secret word is not here."""

    pattern: str                    # e.g. "h_ll_"; MASK for unknown positions
    guessed: frozenset[str]         # every letter tried so far
    correct: frozenset[str]         # guesses that hit
    wrong: frozenset[str]           # guesses that missed
    lives_left: int                 # wrong guesses still allowed; 0 also means
                                    # the game is already lost and being played
                                    # out to measure how many misses it takes
    max_wrong: int
    alphabet: tuple[str, ...]
    candidates: tuple[str, ...]     # every word of this length, unfiltered

    @property
    def length(self) -> int:
        return len(self.pattern)

    @property
    def unguessed(self) -> tuple[str, ...]:
        """Alphabet letters not yet tried, in alphabet order."""
        return tuple(c for c in self.alphabet if c not in self.guessed)

    def matches(self, word: str) -> bool:
        """Is ``word`` still consistent with the pattern and the guesses made?

        A word matches when the revealed letters sit exactly where the pattern
        shows them, no masked position holds an already-guessed letter, and no
        wrongly-guessed letter appears at all.
        """
        if len(word) != len(self.pattern):
            return False
        for want, have in zip(self.pattern, word):
            if want == MASK:
                if have in self.guessed:
                    return False
            elif want != have:
                return False
        return True

    def consistent_candidates(self) -> tuple[str, ...]:
        """Convenience: ``candidates`` filtered through :meth:`matches`."""
        return tuple(w for w in self.candidates if self.matches(w))


class StrategyError(Exception):
    """A strategy returned something that is not a legal guess."""


Strategy = Callable[[GameState], str]


# --------------------------------------------------------------------------- #
# Playing
# --------------------------------------------------------------------------- #

@dataclass
class GameResult:
    """One game, played until the word is fully revealed.

    A lost game does not stop at the lives limit: it keeps going so we can see
    how many misses the word would really have cost.  The lists therefore cover
    the whole play-out, while ``n_guesses``/``n_wrong`` report the game as the
    lives limit would have ended it.
    """

    word: str
    won: bool
    guesses: list[str]              # in order, hits and misses alike
    wrong: list[str]                # the misses, in order
    patterns: list[str]             # pattern after each guess
    max_wrong: int = 6
    stopped_at: int | None = None   # guesses made when the lives ran out

    @property
    def n_guesses(self) -> int:
        """Guesses made before the lives limit ended the game."""
        return len(self.guesses) if self.stopped_at is None else self.stopped_at

    @property
    def n_wrong(self) -> int:
        """Misses inside the lives limit; never more than ``max_wrong``."""
        return min(len(self.wrong), self.max_wrong)

    @property
    def n_misses(self) -> int:
        """Misses needed to reveal the whole word, ignoring the lives limit."""
        return len(self.wrong)

    def __str__(self) -> str:
        head = "WON " if self.won else "LOST"
        tail = "" if self.won else f", {self.n_misses} to finish"
        return (f"{head} {self.word!r} in {self.n_guesses} guesses "
                f"({self.n_wrong} wrong: {''.join(self.wrong[:self.n_wrong])}{tail})")


def play(
    word: str,
    strategy: Strategy,
    words: dict[int, tuple[str, ...]] | None = None,
    max_wrong: int = 6,
    fold_umlauts: bool = False,
    verbose: bool = False,
) -> GameResult:
    """Play one game of hangman against ``word`` using ``strategy``.

    The loop runs until every letter is revealed, even once ``max_wrong``
    misses have been made, so the result can report how many misses the word
    costs in total.  Whether the game counts as won is decided at the moment
    the lives run out; play-out only adds information.  A strategy therefore
    sees states a normal game never reaches -- ``lives_left`` at 0, few letters
    left in the alphabet, possibly no consistent candidates.
    """
    word = fold(word.lower()) if fold_umlauts else word.lower()
    alphabet = FOLDED_ALPHABET if fold_umlauts else ALPHABET

    unknown = set(word)
    if not unknown <= set(alphabet):
        raise ValueError(f"{word!r} contains letters outside the alphabet")

    if words is None:
        words = load_words(fold_umlauts=fold_umlauts)
    candidates = words.get(len(word), ())

    guessed: set[str] = set()
    correct: set[str] = set()
    wrong: list[str] = []
    guesses: list[str] = []
    patterns: list[str] = []

    stopped_at: int | None = None   # guesses made when the lives ran out

    while unknown:
        state = GameState(
            pattern="".join(c if c in correct else MASK for c in word),
            guessed=frozenset(guessed),
            correct=frozenset(correct),
            wrong=frozenset(wrong),
            lives_left=max(0, max_wrong - len(wrong)),
            max_wrong=max_wrong,
            alphabet=alphabet,
            candidates=candidates,
        )

        guess = strategy(state)
        # A strategy bug should crash, not masquerade as a wrong guess.
        if not isinstance(guess, str) or len(guess) != 1:
            raise StrategyError(f"strategy returned {guess!r}, expected a single letter")
        if guess not in alphabet:
            raise StrategyError(f"strategy guessed {guess!r}, which is not in the alphabet")
        if guess in guessed:
            raise StrategyError(f"strategy guessed {guess!r} again (already tried: {''.join(sorted(guessed))})")

        guessed.add(guess)
        guesses.append(guess)
        if guess in unknown:
            unknown.discard(guess)
            correct.add(guess)
        else:
            wrong.append(guess)
        patterns.append("".join(c if c in correct else MASK for c in word))

        # The miss that uses up the last life ends the game proper; everything
        # after it is play-out.  A miss never reveals a letter, so the word is
        # always still incomplete here -- reaching this point means a loss.
        if stopped_at is None and len(wrong) >= max_wrong:
            stopped_at = len(guesses)

        if verbose:
            hit = "hit " if guess in correct else "miss"
            lives = max(0, max_wrong - len(wrong))
            over = "  (played out)" if stopped_at is not None else ""
            print(f"  {guess} {hit}  {patterns[-1]}  lives {lives}{over}")

    return GameResult(word=word, won=stopped_at is None, guesses=guesses,
                      wrong=wrong, patterns=patterns, max_wrong=max_wrong,
                      stopped_at=stopped_at)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

@dataclass
class Stats:
    n_games: int
    n_won: int
    results: list[GameResult]

    @property
    def win_rate(self) -> float:
        return self.n_won / self.n_games if self.n_games else 0.0

    @property
    def avg_guesses(self) -> float:
        return sum(r.n_guesses for r in self.results) / len(self.results) if self.results else 0.0

    @property
    def avg_wrong(self) -> float:
        """Mean misses inside the lives limit; capped at ``max_wrong`` per game."""
        return sum(r.n_wrong for r in self.results) / len(self.results) if self.results else 0.0

    @property
    def avg_misses(self) -> float:
        """Mean misses needed to reveal the whole word, ignoring the lives limit.

        Unlike :attr:`avg_wrong` this is not capped, so a strategy that loses
        badly is not flattered by the games it loses.
        """
        return sum(r.n_misses for r in self.results) / len(self.results) if self.results else 0.0

    @property
    def by_length(self):
        """Per-word-length breakdown, indexed by length.

        Columns mirror the headline numbers in :meth:`__str__`: ``games``,
        ``wins``, ``win_rate``, ``avg_guesses``, ``avg_wrong``, ``avg_misses``.
        Short words are much harder than long ones (fewer letters to reveal,
        many more candidates per pattern), so a single win rate hides most of
        what separates two strategies.
        """
        import pandas as pd

        rows = [{"length": len(r.word), "won": r.won, "guesses": r.n_guesses,
                 "wrong": r.n_wrong, "misses": r.n_misses} for r in self.results]
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        out = df.groupby("length").agg(games=("won", "size"),
                                       wins=("won", "sum"),
                                       avg_guesses=("guesses", "mean"),
                                       avg_wrong=("wrong", "mean"),
                                       avg_misses=("misses", "mean"))
        out["win_rate"] = out["wins"] / out["games"]
        return out[["games", "wins", "win_rate", "avg_guesses", "avg_wrong", "avg_misses"]]

    @property
    def loss_letter_freq(self) -> dict[str, float]:
        """Relative by-word letter frequency over the words that were lost.

        Counts each letter once per lost word (see :func:`letter_frequency`),
        so it answers "which letters show up in the words this strategy can't
        finish?" rather than "which letters did it waste guesses on".  Compare
        against :func:`letter_frequency` over the whole word list to see which
        letters are over-represented in the failures.  ``{}`` when nothing was
        lost.
        """
        return letter_frequency((r.word for r in self.losses()), by_word=True)

    @property
    def loss_lengths(self):
        """Number of lost words per word length, indexed by length.

        Raw counts, so they are confounded by how many games each length got:
        :func:`sample_words` draws ``n_per_length`` words per bucket *except*
        where the bucket holds fewer (length 1 has 26 words, length 2 has 87),
        and the longest lengths only a handful.  For a normalized view use
        ``by_length.win_rate``.
        """
        import pandas as pd

        lengths = [len(r.word) for r in self.losses()]
        counts = pd.Series(lengths, dtype=int).value_counts().sort_index()
        return counts.rename("losses").rename_axis("length")

    def losses(self) -> list[GameResult]:
        return [r for r in self.results if not r.won]

    def __str__(self) -> str:
        return (f"{self.n_won}/{self.n_games} won ({self.win_rate:.1%})  "
                f"avg guesses {self.avg_guesses:.2f}  avg wrong {self.avg_wrong:.2f}  "
                f"avg misses {self.avg_misses:.2f}")


def sample_words(
    words: dict[int, tuple[str, ...]],
    n_per_length: int,
    seed: int = 0,
    min_length: int = 3,
    max_length: int | None = None,
) -> list[str]:
    """Draw ``n_per_length`` words from *every* length bucket in range.

    Sampling per length rather than from one pooled list stops the crowded
    middle lengths from swamping the short words, which are the hard ones --
    every row of :attr:`Stats.by_length` then rests on the same number of
    games.  A bucket holding fewer words than asked for contributes all of
    them, so the long tail (one 85-letter noun exists) adds a game each.
    Reproducible for a given ``seed`` and bounds.
    """
    rng = random.Random(seed)
    picked: list[str] = []
    for length, ws in sorted(words.items()):
        if length < min_length or (max_length is not None and length > max_length):
            continue
        picked.extend(rng.sample(ws, min(n_per_length, len(ws))))
    return picked


def evaluate(
    strategy: Strategy,
    words: dict[int, tuple[str, ...]] | None = None,
    n_games: int = 50,              # per word length, not in total
    seed: int = 0,
    max_wrong: int = 6,
    min_length: int = 3,
    max_length: int | None = None,
    test_words: Sequence[str] | None = None,
    fold_umlauts: bool = False,
) -> Stats:
    """Run ``strategy`` over a seeded sample of words and collect statistics.

    ``n_games`` is **per word length**: each length bucket between
    ``min_length`` and ``max_length`` contributes that many words (or all of
    them, if it holds fewer), so ``Stats.n_games`` is the total across
    lengths and comes out well above ``n_games``.  With the default bounds
    ``n_games=500`` is about 10.5k games; keep it small, or cap
    ``max_length``, while iterating.

    The same ``seed`` and bounds give the same words every time, so two
    strategies can be compared on identical games.
    """
    if words is None:
        words = load_words(fold_umlauts=fold_umlauts)
    if test_words is None:
        test_words = sample_words(words, n_games, seed=seed,
                                  min_length=min_length, max_length=max_length)

    results = [play(w, strategy, words, max_wrong=max_wrong, fold_umlauts=fold_umlauts)
               for w in test_words]
    return Stats(n_games=len(results), n_won=sum(r.won for r in results), results=results)


def compare(strategies: dict[str, Strategy], **kwargs):
    """Evaluate several strategies on the same games; returns a summary DataFrame.

    ``avg_wrong`` counts misses inside the lives limit, ``avg_misses`` the
    misses needed to reveal the whole word with the limit ignored.
    """
    import pandas as pd

    rows = []
    for name, strat in strategies.items():
        s = evaluate(strat, **kwargs)
        rows.append({"strategy": name, "games": s.n_games, "win_rate": s.win_rate,
                     "avg_guesses": s.avg_guesses, "avg_wrong": s.avg_wrong,
                     "avg_misses": s.avg_misses})
    return pd.DataFrame(rows).set_index("strategy")


# --------------------------------------------------------------------------- #
# Baseline strategy: fixed frequency order, ignores the pattern entirely.
# This exists to prove the harness works and to give you a number to beat.
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=None)
def _frequency_order(fold_umlauts: bool = False) -> tuple[str, ...]:
    words = load_words(fold_umlauts=fold_umlauts)
    freq = letter_frequency((w for ws in words.values() for w in ws), by_word=True)
    return tuple(freq)


def frequency_strategy(state: GameState) -> str:
    """Guess letters in global by-word frequency order, ignoring the board."""
    folded = "ä" not in state.alphabet
    for c in _frequency_order(folded):
        if c not in state.guessed:
            return c
    return state.unguessed[0]


if __name__ == "__main__":
    words = load_words()
    print(f"{sum(len(v) for v in words.values())} words, "
          f"lengths {min(words)}-{max(words)}")
    stats = evaluate(frequency_strategy, words, n_games=50, seed=0)
    print("frequency_strategy:", stats)
