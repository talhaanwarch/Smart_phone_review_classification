# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 00:33:24 2020

@author: TAC
"""


# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 10:46:18 2020

@author: TAC
"""

import numpy as np
import pandas as pd
import nltk
import string
import joblib
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,make_scorer,f1_score,recall_score,precision_score
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
nltk.download('wordnet')
nltk.download('stopwords')
from sklearn.model_selection import cross_validate
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize,WhitespaceTokenizer
from nltk.corpus import stopwords



#preprocessing
lemmatizer=WordNetLemmatizer()
tokenizer=WhitespaceTokenizer()
stop_words = set(stopwords.words("english")) 

def lemmatization(text):
    """This function remove stop words, convert to lower and then lemmatize the sentence"""
    return ' '.join([lemmatizer.lemmatize(w.lower()) for w in tokenizer.tokenize(text) if w not in stop_words])
 
def preprocessing(text):
    text=text.str.replace('\d+',' ') #remove all digits from the text
    text=text.apply(lemmatization) #lemmatize 
    text=text.apply(lambda x:x.translate(str.maketrans('', '', string.punctuation))) #remove punctuation
    return text


def ngram_extraction(test_data):
    #vectorizer=CountVectorizer(analyzer='word',max_features=1000,ngram_range=(ngram),max_df=1.0,min_df=ndocs)
    vectorizer = joblib.load('uni_vec.pkl')
    preds=vectorizer.transform(test_data)
    return preds

def classification(vecotr):
    clf=joblib.load('uni_rf.pkl')
    y_pred=clf.predict(vecotr)
    return y_pred






def main():
    # load data
    
    test=pd.read_csv('test_reviews.csv')
    test_text=test['extract']
    test_text=preprocessing(test_text)
    preds=ngram_extraction(test_text)
    y_preds=classification(preds)
    y_pred=pd.DataFrame(y_preds)
    y_pred.to_csv('predicted_rating.csv', index=False)

    



if __name__ == "__main__":
    main()







