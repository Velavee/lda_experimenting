# https://towardsdatascience.com/topic-modelling-in-python-with-nltk-and-gensim-4ef03213cd21
import re
import random
import pickle

import gensim
from gensim import corpora

from text_processing import *


def main():
    text_data = []
    with open("dataset.csv") as f: # Most examples seem to use .csv. Why?
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

if __name__ == '__main__':
    main()
