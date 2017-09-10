import nltk
import numpy
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.corpus import movie_reviews
import random
from sklearn.naive_bayes import MultinomialNB,GaussianNB,BernoulliNB
import pickle


document=[(list(movie_reviews.words(fileid)),category)
          for category in movie_reviews.categories()
          for fileid in movie_reviews.fileids(category)]
random.shuffle(document)
all_words=[]
for w in movie_reviews.words():
    all_words.append(w.lower())
all_words=nltk.FreqDist(all_words)
word_feature=list(all_words)[:3000]

def find_feature(documents):
    words=set(documents)
    feature={}
    for w in word_feature:
        feature[w]=(w in words)
    return feature
print((find_feature(movie_reviews.words('neg/cv000_29416.txt'))))
features=[(find_feature(rev),category) for(rev,category) in document]

training_Set=features[:1900]
test_set=features[1900:]
classifier=nltk.NaiveBayesClassifier.train(training_Set)
print("naive bayes accuarac",(nltk.classify.accuracy(classifier,test_set))*100)
