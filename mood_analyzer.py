# mood_analyzer.py
"""
Rule based mood analyzer for short text snippets.

This class starts with very simple logic:
  - Preprocess the text
  - Look for positive and negative words
  - Compute a numeric score
  - Convert that score into a mood label
"""

from typing import List, Dict, Tuple, Optional

from dataset import POSITIVE_WORDS, NEGATIVE_WORDS


class MoodAnalyzer:
    """
    A very simple, rule based mood classifier.
    """

    def __init__(
        self,
        positive_words: Optional[List[str]] = None,
        negative_words: Optional[List[str]] = None,
    ) -> None:
        # Use the default lists from dataset.py if none are provided.
        positive_words = positive_words if positive_words is not None else POSITIVE_WORDS
        negative_words = negative_words if negative_words is not None else NEGATIVE_WORDS

        # Store as sets for faster lookup.
        self.positive_words = set(w.lower() for w in positive_words)
        self.negative_words = set(w.lower() for w in negative_words)

    # ---------------------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------------------

    def preprocess(self, text: str) -> List[str]:
        """
        Convert raw text into a list of tokens the model can work with.

        TODO: Improve this method.

        Right now, it does the minimum:
          - Strips leading and trailing whitespace
          - Converts everything to lowercase
          - Splits on spaces

        Ideas to improve:
          - Remove punctuation
          - Handle simple emojis separately (":)", ":-(", "🥲", "😂")
          - Normalize repeated characters ("soooo" -> "soo")
        """
        cleaned = text.strip().lower()
        tokens = cleaned.split()

        return tokens

    # ---------------------------------------------------------------------
    # Scoring logic
    # ---------------------------------------------------------------------

    def score_text(self, text: str) -> int:
        """
        Compute a numeric "mood score" for the given text.

        Positive words increase the score.
        Negative words decrease the score.

        TODO: You must choose AT LEAST ONE modeling improvement to implement.
        For example:
          - Handle simple negation such as "not happy" or "not bad"
          - Count how many times each word appears instead of just presence
          - Give some words higher weights than others (for example "hate" < "annoyed")
          - Treat emojis or slang (":)", "lol", "💀") as strong signals
        """
        # Implemented by delegating to the analysis helper and returning the score.
        score, _pos, _neg, _negated = self._analyze_text(text)
        return score

    def _analyze_text(self, text: str) -> Tuple[int, List[str], List[str], List[str]]:
        """
        Analyze the text and return (score, positive_hits, negative_hits, negated_hits).

        Enhancement implemented: simple negation handling. If a negation token
        ("not", "never", "no", "n't") appears immediately before a sentiment
        word, the sentiment of that word is inverted.
        """
        tokens = self.preprocess(text)

        negation_words = {"not", "never", "no", "n't"}

        score = 0
        positive_hits: List[str] = []
        negative_hits: List[str] = []
        negated_hits: List[str] = []

        i = 0
        while i < len(tokens):
          token = tokens[i]
          is_negated = False

          # simple look-ahead: if current token is a negation and there's a next
          # token, treat the next token as negated.
          if token in negation_words and i + 1 < len(tokens):
            i += 1
            token = tokens[i]
            is_negated = True

          if token in self.positive_words:
            if is_negated:
              negated_hits.append(token)
              negative_hits.append(token)
              score -= 1
            else:
              positive_hits.append(token)
              score += 1

          elif token in self.negative_words:
            if is_negated:
              negated_hits.append(token)
              positive_hits.append(token)
              score += 1
            else:
              negative_hits.append(token)
              score -= 1

          i += 1

        return score, positive_hits, negative_hits, negated_hits

    # ---------------------------------------------------------------------
    # Label prediction
    # ---------------------------------------------------------------------

    def predict_label(self, text: str) -> str:
        """
        Turn the numeric score for a piece of text into a mood label.

        The default mapping is:
          - score > 0  -> "positive"
          - score < 0  -> "negative"
          - score == 0 -> "neutral"

        TODO: You can adjust this mapping if it makes sense for your model.
        For example:
          - Use different thresholds (for example score >= 2 to be "positive")
          - Add a "mixed" label for scores close to zero
        Just remember that whatever labels you return should match the labels
        you use in TRUE_LABELS in dataset.py if you care about accuracy.
        """
        score = self.score_text(text)

        if score > 0:
          return "positive"
        if score < 0:
          return "negative"
        return "neutral"

    # ---------------------------------------------------------------------
    # Explanations (optional but recommended)
    # ---------------------------------------------------------------------

    def explain(self, text: str) -> str:
        """
        Return a short string explaining WHY the model chose its label.

        TODO:
          - Look at the tokens and identify which ones counted as positive
            and which ones counted as negative.
          - Show the final score.
          - Return a short human readable explanation.

        Example explanation (your exact wording can be different):
          'Score = 2 (positive words: ["love", "great"]; negative words: [])'

        The current implementation is a placeholder so the code runs even
        before you implement it.
        """
        score, positive_hits, negative_hits, negated_hits = self._analyze_text(text)

        return (
          f"Score = {score} "
          f"(positive: {positive_hits or '[]'}, "
          f"negative: {negative_hits or '[]'}, "
          f"negated: {negated_hits or '[]'})"
        )
