import pandas as pd
import numpy as np
import re, nltk
import random
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
#nltk.download('punkt')
from tkinter import *
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


stemmer = PorterStemmer()
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


def tokenize(text):
    # remove non letters
    text = re.sub("[^a-zA-Z]", " ", text)
    # tokenize
    tokens = nltk.word_tokenize(text)
    # stem
    stems = stem_tokens(tokens, stemmer)
    return stems

def init():

    vectorizer = CountVectorizer(
            analyzer='word',
            tokenizer=tokenize,
            lowercase=True,
            stop_words='english',
            max_features=85
        )

    train_data_f = "G:\\py\\data-visualization\\sentiment-analysis\\sampledata.txt"
    test_data_f = "G:\\py\\data-visualization\\sentiment-analysis\\testdata.txt"
    testsize = 827


    #train_d = pd.read_csv(train_data_f, header=None, delimiter="\t", quoting=3, names=["Sentiment", "Text"])

    train_d = pd.read_csv(train_data_f,delimiter="\t", header=None, quoting=3,names=["Sentiment", "Text"])
    #train_d.columns = ["Sentiment", "Text"]
    #print(train_d)

    #test_d = pd.read_csv(test_data_f, header=None, delimiter="\n", quoting=1, names=["Text"])

    test_d = pd.read_csv(test_data_f, delimiter="\t",header=None, quoting=1, names=["Text"])
    #test_d.columns = ["Text"]

    #test_d  = pd.DataFrame(data=pd.read_csv(test_data_f,header=None, delimiter="\n", quoting=1), columns=(['Text']))

        # print test_d.head()
        # print train_d.Sentiment.value_counts()

    print( train_d.shape )

        # np.mean([len(s.split(" ")) for s in train_d.Text])

    corp_data_features = vectorizer.fit_transform(train_d.Text.tolist() + test_d.Text.tolist())

    corp_data_features_nd = corp_data_features.toarray()
        # print corp_data_features_nd.shape
    vocab = vectorizer.get_feature_names()

    dist = np.sum(corp_data_features_nd, axis=0)

        # For each, print the vocabulary word and the number of times it
        # appears in the data set
        # for tag,count in zip(vocab,dist):
        #   print count, tag

    X_train, X_test, y_train, y_test = train_test_split(
            corp_data_features_nd[0:len(train_d)],
            train_d.Sentiment,
            train_size=0.85,
            random_state=1234)

        # nb = GaussianNB()
        # nb = nb.fit(X=X_train,y=y_train)

        # y_pred = nb.predict(X_test)
        # print(classification_report(y_test,y_pred))

    # nb = GaussianNB()
    nb = SVC()
    nb = nb.fit(X=corp_data_features_nd[0:len(train_d)], y=train_d.Sentiment)

    test_pred = nb.predict(corp_data_features_nd[len(train_d):])

    sample = random.sample(range(len(test_pred)), 825)
        # print test_pred
    count = 0

    # for text, sentiment in zip(test_d.Text[sample], test_pred[sample]):
    #     print(sentiment, text)
    print("\n \n \n")
    for i in range(1, 825):
        if (test_pred[i] == 1):
            count = count + 1

    print("Total number of reviews: %s" %testsize)
    if count > 413:
        
        print("Number of positive reviews: %s " %count)
        print("Reviews are positive.")
        return count
    else:
        x = testsize-count
        print("Number of negative reviews: %s " % x)
        print("Reviews are NEGATIVE.")
        return count
        

init()