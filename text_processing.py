import nltk
from nltk.corpus import wordnet as wn
nltk.download('wordnet')
nltk.download('stopwords')
en_stop = set(nltk.corpus.stopwords.words('english'))

import spacy
nlp = spacy.load("en_core_web_sm")

def tokenize(text):
    lda_tokens = []
    tokens = nlp(text) # What processing does this do?
    for token in tokens:
        if token.orth_.isspace() or token.orth_.isdigit():
            continue
        elif token.like_url or token.orth_.startswith("www"):
            lda_tokens.append('URL')
        elif token.orth_.startswith('@'):
            lda_tokens.append('SCREEN_NAME')
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens

def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma

def prepare_text_for_lda(text):
    tokens = tokenize(text)
    tokens = [token for token in tokens if len(token) > 4] # Why do we have this? Aren't there meanigful words that are fewer than 3 letters?
    tokens = [token for token in tokens if token not in en_stop]
    tokens = [get_lemma(token) for token in tokens]
    return tokens