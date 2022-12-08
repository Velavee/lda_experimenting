# https://towardsdatascience.com/topic-modelling-in-python-with-nltk-and-gensim-4ef03213cd21
import spacy
import re
import random
import nltk
import gensim
from gensim import corpora

import pickle

nltk.download('omw-1.4')
from nltk.corpus import wordnet as wn

nltk.download('stopwords')
en_stop = set(nltk.corpus.stopwords.words('english'))

nlp = spacy.load("en_core_web_sm")

def tokenize(text):
    lda_tokens = []
    tokens = nlp(text)
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

text_data = []
with open("there-will-come-soft-rains.txt") as f: # Most examples seem to use .csv. Why?
    for line in f:
        if line != '\n':
            line = re.sub(r'[^\w\s]', '', line)
            tokens = prepare_text_for_lda(line)
            print(tokens)
            text_data.append(tokens)
            # if random.random() > .99: # What is the purpose of this?
                # print(tokens)
                # text_data.append(tokens)

    f.close()

dictionary = corpora.Dictionary(text_data)
corpus = [dictionary.doc2bow(text) for text in text_data]
pickle.dump(corpus, open('corpus.pkl', 'wb')) # Where is this corpus from? Can I use other corpora?
dictionary.save('dictionary.gensim')

NUM_TOPICS = 3 # Does altering the number of topics change the relevance of results?
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=15)
ldamodel.save('model5.gensim')

topics = ldamodel.print_topics(num_words=4)
for topic in topics:
    print(topic)

# Visualization - wasn't working
# import pyLDAvis.gensim
# lda_display = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary, sort_topics=False)
# pyLDAvis.display(lda_display)
