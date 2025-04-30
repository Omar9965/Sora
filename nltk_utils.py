import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from string import punctuation
from nltk.corpus import stopwords



wordnet_lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def tokenize(sentence):
    tokens = nltk.word_tokenize(sentence)
    return [w.lower() for w in tokens if w.lower() not in punctuation and w.lower() not in stop_words]

def lemmatize_word(word):
    return wordnet_lemmatizer.lemmatize(word.lower())

def sentence_to_tfidf(tokenized_sentence, tfidf_vectorizer):
    lemmatized = [lemmatize_word(w) for w in tokenized_sentence]
    sentence_str = " ".join(lemmatized)
    tfidf_vector = tfidf_vectorizer.transform([sentence_str]).toarray()[0]
    return tfidf_vector.astype(np.float32)


