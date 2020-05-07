# Dialogue Act Tagging
Implementing two dialogue act taggers using Conditional Random Field.
### Baseline tagger features:
- Speaker change indicator.
- First utterance indicator.
- A feature for each tokens in the utterance.
- A feature for each part-of-speech in the utterance.

### Advanced Tagger:
- Length of utterance.
- Bigrams of tokens.
- various features extracted from tokens string: e.g. IS_UPPER to indicate whether the token is in upper case.
- A feature for each token and a feature for each POS, like in the baseline.
- Non-words sounds in the utterance, e.g. \<Laughter>
- Non-words sounds in the previous utterance, e.g. \<Laughter>
- Speaker change indicator: as in the baseline.
- A feature for each part-of-speech in the previous utterance.
- A Bias feature.

## Data set
The Switchboard (SWBD), which is a collection of phone dialogues of volunteers over a predetermine topics, is used.
The tags in the corpus are the SWBD-DAMSL dialogue acts. 
See [this](https://web.stanford.edu/~jurafsky/ws97/manual.august1.html) for the annotation manual.


## Conditional Random Field
A python interface of the CRFsuite is used, see [this](https://pypi.python.org/pypi/python-crfsuite) for installation and documentation.

### Credit
`hw2_corpus_tool.py` is written by Christopher Wienberg, a previous TA for CSCI544 at USC.