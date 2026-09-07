## Analysing Hangman

 This repo is the
code behind the writeup
[Solving (german) Hangman with Decision Trees](https://oechsner.dev/blog/analysing-hangman)
Read that first for the analysis and the plots; this is the harness the numbers
came out of.

The setup: the guesser knows the word length, gets 6 wrong guesses, and has
perfect knowledge of the wordset. The secret word is a German noun.

### Data

`nouns.csv` is the **German-Nouns** dump (noun lemmas from German Wikipedia,
~250 inflection columns of which only `lemma` is used). After keeping plain letter-words, lowercasing and
deduplicating: **95,031 nouns, lengths 1–85**, bucketed by length. Lengths form
a bell curve peaking around 10 characters.

### The Core: `hangman.py`

A strategy is just a function `GameState -> str` returning one not-yet-guessed
letter. It never sees the secret word.

```python
import hangman
words = hangman.load_words()                      # {length: (word, ...)}, cached
hangman.play("hangman", my_strategy, words, verbose=True)   # one game, move by move
hangman.evaluate(my_strategy, words, n_games=50)  # stats over a seeded sample
hangman.compare({"a": strat_a, "b": strat_b}, words=words, n_games=50)
```

`GameState` gives you `pattern`, `guessed` / `correct` / `wrong`, `lives_left`,
the `alphabet`, and `candidates` (every word of this length, unfiltered), plus
`state.matches(word)` and `state.consistent_candidates()`

### Running it

```
python hangman.py            # word count + the frequency baseline
```

`wordlist.ipynb` has the data prep, the strategies, and the plots.
`hangman.py` needs `pandas`; the notebook additionally needs `matplotlib`.
