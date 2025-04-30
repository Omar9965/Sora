import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import json
from nltk_utils import tokenize, lemmatize_word, sentence_to_tfidf

# Load intents
with open('Intent.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent["tag"]
    tags.append(tag)
    for pattern in intent['patterns']:
        tokens = tokenize(pattern)
        all_words.extend(tokens)
        xy.append((tokens, tag))

all_words = [lemmatize_word(w) for w in all_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))


lemmatized_sentences = [" ".join([lemmatize_word(w) for w in pattern]) for pattern, _ in xy]
tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit(lemmatized_sentences)

X = []
y = []
for (pattern_sentence, tag) in xy:
    tfidf = sentence_to_tfidf(pattern_sentence, tfidf_vectorizer)
    X.append(tfidf)
    y.append(tags.index(tag))

X = np.array(X)
y = np.array(y)