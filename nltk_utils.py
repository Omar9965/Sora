import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from string import punctuation
import numpy as np
wordnet_lemmatizer = WordNetLemmatizer()


def tokenize(sentence):
    tokens = nltk.word_tokenize(sentence)
    return [w for w in tokens if w not in punctuation]





def lemmatize_word(word):

    return wordnet_lemmatizer.lemmatize(word.lower())





def sentence_to_tfidf(tokenized_sentence, tfidf_vectorizer):
    lemmatized = [lemmatize_word(w) for w in tokenized_sentence]
    sentence_str = " ".join(lemmatized)
    tfidf_vector = tfidf_vectorizer.transform([sentence_str]).toarray()[0]
    return tfidf_vector.astype(np.float32)