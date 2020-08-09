# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 01:09:23 2020

@author: TAC
"""



import re, nltk
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix,accuracy_score,f1_score,precision_score,recall_score
from sklearn.ensemble import AdaBoostClassifier
import joblib




def normalizer(text): #### Cleaning Tweets    
    re2 = re.sub("[^A-Za-z]+"," ", text) # removing numbers
    tokens = nltk.word_tokenize(re2)
    removed_letters = [word for word in tokens if len(word)>2] # removing words
    lower_case = [l.lower() for l in removed_letters]
    stop_words = set(stopwords.words('english'))
    filtered_result = list(filter(lambda l: l not in stop_words, lower_case))
    wordnet_lemmatizer = WordNetLemmatizer()
    lemmas = ' '.join([wordnet_lemmatizer.lemmatize(t, pos='v') for t in filtered_result])
    return lemmas





    
def classification(text):
    vectorizer = joblib.load('tfidf.pkl')
    vec_pred=vectorizer.transform(text)
    clf=joblib.load('svm.pkl')
    clf_pred=clf.predict(vec_pred)
    return clf_pred

    
def main():
    data=pd.read_csv('test_reviews.csv',usecols=['extract'],squeeze=True)
    text=data.apply(normalizer)
    y_pred=classification(text)
    y_pred=pd.DataFrame(y_pred)
    y_pred.to_csv('predicted_rating.csv',  encoding='utf-8',index=False)
    
    
if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    return train_vector,val_vector