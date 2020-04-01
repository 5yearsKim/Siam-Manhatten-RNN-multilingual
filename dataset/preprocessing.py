import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.stem.snowball import SpanishStemmer
from nltk.tokenize.toktok import ToktokTokenizer
import pandas as pd



# Data Preprocessing
def en_preprocessing(sent):
    # tokenize into words
    tokens = [word for word in nltk.word_tokenize(sent)]

    # remove stopwords and make it lower case
    stop = stopwords.words('english')
    tokens = [token.lower() for token in tokens if token not in stop]

    # remove words less than two letters
    tokens = [word for word in tokens if len(word) >= 2]

    # lemmatization
    lmtzr = WordNetLemmatizer()
    tokens = [lmtzr.lemmatize(word) for word in tokens]
    tokens = [lmtzr.lemmatize(word, 'v') for word in tokens]

    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return tokens


def es_preprocessing(sent):
    # tokenize into words
    toktok = ToktokTokenizer()
    tokens = [word for word in toktok.tokenize(sent)]

    # remove stopwords and make it lower case
    stop = stopwords.words('spanish')
    tokens = [token.lower() for token in tokens if token not in stop]

    # remove words less than two letters
    tokens = [word for word in tokens if len(word) >= 2]

    stemmer = SpanishStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return tokens


def process_pair_data():
    readpath = "./dataset/XNLI-15way/en_es_siam.csv"
    writepath = "./dataset/XNLI-15way/en_es_siam_processed.csv"

    df = pd.read_csv(readpath, sep=',', index_col=0)

    word2index = {"UKN":0}

    for i in range(df.shape[0]):

        en_tokens = en_preprocessing(df.loc[i, 'en'])
        es_tokens = es_preprocessing(df.loc[i, 'es'])

        for token in en_tokens:
            if token not in word2index:
                word2index[token] = len(word2index)
        for token in es_tokens:
            if token not in word2index:
                word2index[token] = len(word2index)

        df.loc[i, 'en'] = " ".join(list(map(lambda x: str(word2index[x]), en_tokens)))
        df.loc[i, 'es'] = " ".join(list(map(lambda x: str(word2index[x]), es_tokens)))

    df.to_csv(writepath)
    return word2index


if __name__ == "__main__":
    import json
    w2i = process_pair_data()
    # Serialize data into file:
    json.dump(w2i, open("word2index.json", 'w'))
