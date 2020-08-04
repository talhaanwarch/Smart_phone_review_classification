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

def label_encoding(labels,binary=True):
    if binary is True:
        print('binary encoding')
        labels[labels<8]=0
        labels[labels>=8]=1
    else:
        print('multi_class encoding')
        encoder=LabelEncoder()
        labels=encoder.fit_transform(labels)
    return labels

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


def ngram_extraction(train_text,test_text,ngram=(1,1),ndocs=10):
    vectorizer=CountVectorizer(analyzer='word',max_features=1000,ngram_range=(ngram),max_df=1.0,min_df=ndocs)
    #max_df is set to 100%, "ignore terms that appear in more than 100% of the documents". 
    #min_df means "ignore terms that appear in less than 10 documents".
    train_vector=vectorizer.fit_transform(train_text)
    test_vector=vectorizer.fit_transform(test_text)
    return train_vector,test_vector





def cross_validation(clf,X,y):
    print('cross validation')
    scoring = {'accuracy' : make_scorer(accuracy_score), 
           'precision' : make_scorer(precision_score,average='macro'),
           'recall' : make_scorer(recall_score,average='macro'), 
           'f1_score' : make_scorer(f1_score,average='macro')}
    return cross_validate(clf, X=X, y=y, cv=5, n_jobs=-1,scoring=scoring)


from sklearn.svm import SVC,LinearSVC
def classification(model,X_train,y_train,X_val,y_val,file,re_train=False):
    clf = model
    if re_train==True:
        clf.fit(X_train,y_train)
        y_pred=clf.predict(X_val)
        accuracy=accuracy_score(y_val,y_pred)
        print(classification_report(y_val,y_pred))
        joblib.dump(clf,file)
    else: 
        clf = joblib.load(file)
        y_pred=clf.predict(X_val)
        accuracy=accuracy_score(y_val,y_pred)
        print(classification_report(y_val,y_pred))
    return accuracy


def save_model(model,data,label,ngram,ndocs,vec_file,clf_file):
    vectorizer=CountVectorizer(analyzer='word',max_features=1000,ngram_range=(ngram),max_df=1.0,min_df=ndocs)
    data=vectorizer.fit_transform(data)
    print('vector saving')
    joblib.dump(vectorizer,vec_file)
    model.fit(data,label)
    print('model saving')
    joblib.dump(model,clf_file)


def main(savemodel=False):
    # load data
    train=pd.read_csv('user_reviews.csv')
    test=pd.read_csv('test_reviews.csv')
    
    train_text=train['extract']
    test_text=test['extract']
    train_labels=train['score'].astype('int')
    
    train_labels=label_encoding(train_labels)
    train_text=preprocessing(train_text)
    test_text=preprocessing(test_text)
    
    #train['score'].value_counts()
    if savemodel is True:
        save_model(RandomForestClassifier(n_jobs=-1),train_text,train_labels,(1,1),10,'uni_vec.pkl','uni_rf.pkl')
        
    
    else:
        train_vector,test_vector=ngram_extraction(train_text,test_text,ngram=(1,3),ndocs=30)
        
        #X_train,X_val,y_train,y_val=train_test_split(train_vector,train_labels,test_size=0.3,random_state=2020)
        
        model=RandomForestClassifier()
        #accuracy=classification(model,X_train,y_train,X_val,y_val,file='binary/SVM_2.pkl',re_train=True)
        result=cross_validation(model,train_vector,train_labels)
        result=pd.DataFrame.from_dict(result)
        result.columns=['fit_time','score_time','accuracy',	'precision'	,'recall',	'f1_score']
        result.loc['mean'] = result.mean()
        result.loc['std'] = result.std()
        result=result.iloc[:,2::]
        result=result.round(3)*100
        result.to_clipboard()
        #print(accuracy)
    


main(savemodel=False)







