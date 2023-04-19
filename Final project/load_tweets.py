import spacy
import random
from nltk.corpus import twitter_samples
import nltk
import time
nltk.download('twitter_samples')

def load_tweets():
    start = time.time()
    print("Loading & lemmatizing tweets...")
    randomState = random.Random(4120)
    pos_tokens = twitter_samples.tokenized('positive_tweets.json')
    neg_tokens = twitter_samples.tokenized('negative_tweets.json')
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    pos_data = [" ".join([token.lemma_ for token in nlp(" ".join(token))]) for token in pos_tokens]
    neg_data = [" ".join([token.lemma_ for token in nlp(" ".join(token))]) for token in neg_tokens]
    pos_data = [(str(i), pos_data[i], '1') for i in range(len(pos_data))]
    neg_data = [(str(i), neg_data[i], '0') for i in range(len(neg_data))]
    data = pos_data + neg_data
    randomState.shuffle(data)
    print("Finished loading & lemmatizing tweets. Time elapsed:", time.time() - start)
    
    return data