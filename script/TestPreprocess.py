# https://www.kaggle.com/lystdo/lstm-with-word2vec-embeddings

import re

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# import sys
# # reload(sys)
# sys.setdefaultencoding('utf-8')

'''
text: [ sentence1, sentence2, ... ]
'''


def text_to_wordlist(text_list, remove_stop_words=False, stem_words=False):
    text_list = [x.lower().split() for x in text_list]

    # remove the stop words
    if remove_stop_words:
        stops = set(stopwords.words("english"))
        text_list = [[w for w in sentence if not w in stops] for sentence in text_list]

    text_list = [(" ").join(sentence) for sentence in text_list]
    text_list = [cleanText(sentence) for sentence in text_list]

    # Optionally, shorten words to their stems
    if stem_words:
        stemmer = SnowballStemmer('english')
        stemmed_words = [[stemmer.stem(word) for word in sentence] for sentence in text_list]
        text_list = [" ".join(words) for words in stemmed_words]

    # return cleand sentence
    return text_list


def cleanText(text):
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    return text