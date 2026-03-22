# Model Card: Mood Machine

This model card is for the Mood Machine project, which includes **two** versions of a mood classifier:

1. A **rule based model** implemented in `mood_analyzer.py`
2. A **machine learning model** implemented in `ml_experiments.py` using scikit learn

You may complete this model card for whichever version you used, or compare both if you explored them.

## 1. Model Overview

**Model type:**  
I explored and compared both models:

- The rule-based model implemented in `mood_analyzer.py` (primary focus for the lab changes).
- A small ML baseline implemented in `ml_experiments.py` using `CountVectorizer` + `LogisticRegression`.

**Intended purpose:**  
Classify short text messages (social posts / chat-like snippets) into mood labels such as `positive`, `negative`, `neutral`, or `mixed`.

**How it works (brief):**  
Rule based version (`mood_analyzer.py`):

- Preprocesses text by trimming, lowercasing, and splitting on whitespace.
- Token-level scoring: start at 0, add +1 for each positive word, subtract -1 for each negative word.
- Simple negation handling: if a negation token (`not`, `never`, `no`, `n't`) appears immediately before a sentiment word, the sentiment of that word is inverted (positive→-1, negative→+1).
- Label mapping: score > 0 → `positive`; score < 0 → `negative`; score == 0 → `neutral`.
- `explain()` returns the numeric score and lists of positive, negative, and negated hits to make decisions transparent.

ML baseline (`ml_experiments.py`):

- Uses `CountVectorizer` to convert texts to bag-of-words counts and trains a `LogisticRegression` classifier on `SAMPLE_POSTS` and `TRUE_LABELS` from `dataset.py`.
- Evaluations in the file use the training set for quick feedback (so reported accuracy is training accuracy, not a held-out test score).



## 2. Data

**Dataset description:**  
The starter dataset (`dataset.py`) contains 13 short posts in `SAMPLE_POSTS` paired with `TRUE_LABELS`.
I did not add additional posts for this iteration; all observations below use those 13 examples.

**Labeling process:**  
Labels are the provided `TRUE_LABELS` for each post. Some examples are intentionally ambiguous or mixed and the starter labels reflect an annotation choice (for example: "Feeling tired but kind of hopeful" → `mixed`).

Posts that are hard to label or could plausibly be labeled differently include:

- "Had an amazing day but also kind of want to cry" — annotated `mixed` because it clearly contains both strong positive (`amazing`) and negative (want to cry) signals.
- "This movie was so bad it was actually funny 😂" — annotated `positive` because the writer expresses enjoyment despite the literal negative word "bad" (this is sarcasm/irony).
- "I absolutely love getting stuck in traffic" — annotated `negative` because this is intended sarcastically (enjoyment is unlikely), but literal positive word `love` misleads simple lexical rules.

**Important characteristics of your dataset:**  

- Contains slang or emojis  
- Includes sarcasm  
- Some posts express mixed feelings  
- Contains short or ambiguous messages
Key characteristics:

- Short, conversational posts (single-sentence social-style updates).
- Contains sarcasm and irony (which flip literal sentiment).
- Mixed-mood examples where positive and negative signals coexist in one post.
- Emojis (e.g., 😂) appear and can change interpretation.

**Possible issues with the dataset:**  
Issues:

- Very small: only 13 labeled examples — too small to reliably train a general ML classifier.
- Label ambiguity: several posts are legitimately `mixed` or ambiguous, which both models struggle with.
- Cultural and lexical coverage is narrow: limited slang and emoji variety, few dialectal or multilingual examples.

## 3. How the Rule Based Model Works (if used)

**Your scoring rules:**  

- How positive and negative words affect score  
- Negation rules you added  
- Weighted words  
- Emoji handling  
- Threshold decisions for labels
Implemented rules (summary of choices in `mood_analyzer.py`):

- Tokenization: simple whitespace-splitting after lowercasing.
- Base scoring: each occurrence of a word in `POSITIVE_WORDS` adds +1; each in `NEGATIVE_WORDS` subtracts 1.
- Negation handling: if a negation token immediately precedes a sentiment token, the word's polarity is inverted and the inverted score is applied (this is a local, one-token lookahead rule).
- Label thresholds: a simple three-way threshold based on sign of score (no margin for `mixed` — the dataset's `mixed` labels are not produced by this labeler, they can appear only if a downstream system maps numeric ranges to `mixed`).

These choices prioritize interpretability and a tight, inspectable rule set over coverage of complex phenomena.

**Strengths of this approach:**  
Strengths:

- Transparent and easy to debug: `explain()` lists which tokens influenced the decision.
- Fast and deterministic — no training required.
- Works well when a post contains unambiguous sentiment words (e.g., "I love this class so much").

**Weaknesses of this approach:**  
Concrete failure modes and examples:

- Sarcasm/irony: "I absolutely love getting stuck in traffic" (true label `negative`). The rule-based model counts `love` as +1 and returns a positive score; the negation logic does not apply since there's no negation token.

- Contrast/masking: "This movie was so bad it was actually funny 😂" (true label `positive`). The literal token `bad` is in `NEGATIVE_WORDS` and will decrement the score; unless the model sees other positive tokens, it may predict `negative` even though the post expresses enjoyment. The emoji and the phrase "actually funny" are strong cues but not covered by the simple lexicon.

- Mixed feelings: "Had an amazing day but also kind of want to cry" (true label `mixed`). The model will add +1 for `amazing` and likely 0 for the rest (unless `cry` or `want` are in the negatives), producing `positive` rather than `mixed`.

- Ambiguous short messages: "This is fine" (annotated `neutral`) — `fine` is not in starter lexicons so the model often returns `neutral`, but similar short phrases can be interpreted sarcastically in context.

In short: the model is lexicon-bound and fails when sentiment is signaled by context, phrasing, or non-lexical cues (emoji, syntax, or world knowledge).

## 4. How the ML Model Works (if used)

**Features used:**  
The ML baseline uses bag-of-words (`CountVectorizer`) features — each text is converted into token counts and fed to `LogisticRegression`.

**Training data:**  
The ML model was trained on the same `SAMPLE_POSTS` and `TRUE_LABELS` from `dataset.py` (13 examples in the starter set).

**Training behavior:**  
With such a small dataset, the ML classifier tends to memorize surface patterns in the training set. It can sometimes correct a rule-based failure if the training labels expose a pattern (for example, mapping the phrase "so bad it was actually funny" to `positive`), but it is highly sensitive to label noise and will overfit easily.

Because the training set is the same as the evaluation set in quick experiments, reported accuracy is optimistic (training accuracy), not a true generalization measure.

**Strengths and weaknesses:**  
Observed differences from the rule-based model:

- The ML model can capture multi-word patterns and distributional cues (e.g., it may learn that the phrase "actually funny" in combination with "bad" signals positive sentiment in this dataset).
- It does not require a manually curated lexicon, so it can learn from examples where literal words are misleading.

However:

- With only 13 examples, the ML model overfits and learns dataset idiosyncrasies rather than robust signals.
- It is sensitive to label choices: changing a single label (e.g., labeling a sarcastic example as `negative` instead of `positive`) can flip the classifier's behavior on similar sentences.
- It is less interpretable than the rule-based model unless additional tooling (feature inspection) is used.

## 5. Evaluation

**How you evaluated the model:**  
Evaluation approach:

- Rule-based: inspected predictions for the 13 starter posts using `mood_analyzer.MoodAnalyzer.predict_label()` and `explain()` to see which tokens contributed.
- ML baseline: trains on `SAMPLE_POSTS` and `TRUE_LABELS` and reports training accuracy using `evaluate_on_dataset()` in `ml_experiments.py` (this prints each prediction and the computed accuracy).

Observed patterns (qualitative):

- The rule-based model correctly handles explicit sentiment and the negation cases we implemented (e.g., "I am not happy about this" → `negative` after negation logic).
- The rule-based model fails on sarcastic lines that use positive words to convey a negative sentiment (e.g., "I absolutely love getting stuck in traffic").
- The ML model sometimes corrects specific lexical traps present in `TRUE_LABELS` because it can learn local patterns, but it does so only when those patterns appear consistently in the tiny training set.

**Examples of correct predictions:**  
- "I love this class so much" → `positive` (rule-based and ML). The token `love` is in `POSITIVE_WORDS`, yielding a positive score.
- "I am not happy about this" → `negative` (rule-based with negation). The negation rule inverts `happy` and yields negative score.

**Examples of incorrect predictions:**  
- "I absolutely love getting stuck in traffic" — true label: `negative`. Rule-based prediction: `positive` (counts `love`). ML prediction: depends on training labels; with the small dataset it often follows the literal token evidence and may predict `positive` unless trained on many sarcastic examples. Root cause: sarcasm/irony not captured by lexicon or local negation.

- "This movie was so bad it was actually funny 😂" — true label: `positive`. Rule-based prediction: `negative` (counts `bad`). ML prediction: more likely to be `positive` if the dataset includes this sentence labeled `positive` so the classifier learns the multi-word pattern ("so bad it was actually funny"). Root cause: literal negative words overridden by idiomatic phrasing and emoji.

- "Had an amazing day but also kind of want to cry" — true label: `mixed`. Rule-based prediction: often `positive` (counts `amazing`), missing the negative subtext. ML model might predict `mixed` only if `mixed` is a learned class and present in training with enough examples; otherwise it picks one label.

## 6. Limitations

Describe the most important limitations.  
Examples:  

- The dataset is small  
- The model does not generalize to longer posts  
- It cannot detect sarcasm reliably  
- It depends heavily on the words you chose or labeled
Specific limitations with examples:

- Small training set: with only 13 examples, the ML model overfits and the learned patterns do not generalize. For example, if you relabel the sarcastic traffic sentence differently, the classifier's behavior on similar sentences changes dramatically.

- Sarcasm and idioms: "I absolutely love getting stuck in traffic" and "This movie was so bad it was actually funny 😂" show how literal lexicon-based scoring fails. Our negation rule does not help because sarcasm lacks explicit negation tokens.

- Mixed-feelings and aggregation: sentences that combine positive and negative signals ("amazing" + "want to cry") are labeled `mixed` by the human annotator, but the rule-based model reduces the content to a single integer sign and therefore commonly returns `positive` or `negative` rather than `mixed`.

- Emoji and punctuation: the current preprocess step only lowercases and splits by whitespace; emojis and punctuation are not specially handled. For instance, the laughing-crying emoji (😂) is a strong positive clue in context but is ignored by token matching unless explicitly added to lexicons or tokenization rules.

- Cultural and lexical bias: the lexicon reflects a limited English register and may not cover slang, dialectal forms, or non-English content. The model is optimized for the kinds of short, social-style English messages in `SAMPLE_POSTS` and will misinterpret language from other communities.

## 7. Ethical Considerations

Discuss any potential impacts of using mood detection in real applications.  
Examples: 

- Misclassifying a message expressing distress  
- Misinterpreting mood for certain language communities  
- Privacy considerations if analyzing personal messages
Potential harms and considerations:

- Misclassifying distress: falsely labeling a message as `neutral` or `positive` when it expresses distress could prevent timely support in moderation or safety applications.
- Cultural bias: the lexicon and small training set reflect particular word choices and idioms; speakers of different dialects or those who use non-standard spelling may be misinterpreted.
- Context sensitivity and privacy: mood classification often depends on context (conversation history, user-specific language). Applying this model to private messages without consent raises privacy concerns.

Mitigations: clearly document scope, use human-in-the-loop review for safety-critical decisions, and collect more diverse, consented training data before deployment.

## 8. Ideas for Improvement

List ways to improve either model.  
Possible directions:  

- Add more labeled data  
- Use TF IDF instead of CountVectorizer  
- Add better preprocessing for emojis or slang  
- Use a small neural network or transformer model  
- Improve the rule based scoring method  
- Add a real test set instead of training accuracy only
Short roadmap for improvements:

- Expand `SAMPLE_POSTS` with 100s of labeled examples covering slang, sarcasm, and emoji variants.
- Improve `preprocess()` in `mood_analyzer.py` to strip punctuation, handle emojis as tokens, and normalize repeated characters.
- Add phrase patterns and weighting to the rule-based scorer (e.g., treat "so X it Y" patterns specially), or extend negation to handle a wider window.
- For ML: use TF-IDF, n-gram features, and a held-out test set; consider lightweight transformer fine-tuning with more data.
- Add an explicit `mixed` threshold or multi-label output so that the rule-based model can express simultaneous positive and negative signals instead of collapsing to a single sign.

If you'd like, I can implement `preprocess()` emoji handling and add 10 diverse example posts to `dataset.py` next.
