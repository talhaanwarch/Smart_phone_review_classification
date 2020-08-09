# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 16:21:16 2020

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




def extract_features(train_text,val_text,ngram,thresh):
    #feature extraction using tf-0idf
    vectorizer=TfidfVectorizer(analyzer='word',max_features=1500,ngram_range=(ngram),max_df=1.0,min_df=thresh)  
    train_vector=vectorizer.fit_transform(train_text) 
    val_vector=vectorizer.transform(val_text)
    return train_vector,val_vector

def label_encode(labels):
    #convert labels to binary 
    labels[labels<8]=0
    labels[labels>=8]=1
    return labels

def cross_validation(df,model,ngram,thresh):
    #apply k fold cross validation
    accuracy,f1,precision,recall=[],[],[],[]
    kf = KFold(n_splits=5, shuffle=True, random_state=1)  #k fold cross validatoon
    for train_idx,val_idx in kf.split(df):
        train=df.iloc[train_idx,:] #slice training data
        val=df.iloc[val_idx,:] #slice validation data
        
        train_text,train_labels=train['extract'].copy(),label_encode(train['score'].copy()) #slice features and labels seprate for training data
        val_text,val_labels=val['extract'].copy(),label_encode(val['score'].copy())#slice features and labels seprate for validation data
        train_vector,val_vector=extract_features(train_text,val_text,ngram,thresh)#feature extraction
        #fit the model
        model.fit(train_vector,train_labels)
        y_pred=model.predict(val_vector)
        y_pred=y_pred.astype('int') #convert float to integers
        accuracy.append(accuracy_score(val_labels,y_pred)) #accuracy
        f1.append(f1_score(val_labels,y_pred,average='macro')) #f1 score
        precision.append(precision_score(val_labels,y_pred,average='macro')) #precison
        recall.append(recall_score(val_labels,y_pred,average='macro')) #recall
    
    #save model resuls in a dataframe
    result=pd.DataFrame(data=list(zip(accuracy,precision,recall,f1)),columns=['Accuracy','Precision','Recall','F1-score'])
    result.loc['Average'] = result.mean() #calculate average of 5 fold cross validation
    return result
        

    
def save_model(df,model,ngram,thresh):
    #save the model
    labels=df['score']
    labels[labels<8]=0 
    labels[labels>=8]=1
    #apply tf idf
    vectorizer=TfidfVectorizer(analyzer='word',max_features=1500,ngram_range=(ngram),max_df=1.0,min_df=thresh)  
    data=vectorizer.fit_transform(df['extract'])
    #save model
    joblib.dump(vectorizer,'vec.pkl')
    model.fit(data,labels)
    joblib.dump(model,'clf.pkl')    


def main():
    df=pd.read_csv('user_reviews.csv') #read file
    df['extract']=df['extract'].apply(normalizer) #apply preporcessing
    model1=MultinomialNB() #naive bayes moels
    res1=cross_validation(df,model1,ngram=(1,1),thresh=10) # naive bayes unigram
    res2=cross_validation(df,model1,ngram=(1,2),thresh=20) # naive bayes bigram 
    res3=cross_validation(df,model1,ngram=(1,3),thresh=30) # naive bayes trigram
    
    #run svm model
    model2=LinearSVC()
    res12=cross_validation(df,model2,ngram=(1,1),thresh=10) #svm unigram
    res22=cross_validation(df,model2,ngram=(1,2),thresh=20) #svm bigram
    res32=cross_validation(df,model2,ngram=(1,3),thresh=30) #svm trigram
    
    #run adda boost model
    model3=AdaBoostClassifier()
    res13=cross_validation(df,model3,ngram=(1,1),thresh=10) #adaboost unigram
    res23=cross_validation(df,model3,ngram=(1,2),thresh=20) #adaboost bigram
    res33=cross_validation(df,model3,ngram=(1,3),thresh=30) # #adaboost trigram
    
    #extract average f1 score of all models
    out1f=res1.loc['Average']['F1-score']
    out2f=res2.loc['Average']['F1-score']
    out3f=res3.loc['Average']['F1-score']
    
    out4f=res12.loc['Average']['F1-score']
    out5f=res22.loc['Average']['F1-score']
    out6f=res32.loc['Average']['F1-score']
    
    out7f=res13.loc['Average']['F1-score']
    out8f=res23.loc['Average']['F1-score']
    out9f=res33.loc['Average']['F1-score']
    
    #create a list containing score of all models
    lis=[out1f,out2f,out3f,out4f,out5f,out6f,out7f,out8f,out9f]
    
    #save the model which has the best score
    if np.argmax(lis)==0:
        save_model(df,model1,(1,1),10)
        
    elif np.argmax(lis)==1:
        save_model(df,model1,(1,2),20)
   
    elif np.argmax(lis)==2:
        save_model(df,model1,(1,3),30)
        
    elif np.argmax(lis)==3:
        save_model(df,model2,(1,1),10)
        
    elif np.argmax(lis)==4:
        save_model(df,model2,(1,2),20)
        
    elif np.argmax(lis)==5:
       save_model(df,model2,(1,1),30)
  
    elif np.argmax(lis)==6:
        save_model(df,model3,(1,1),10)
 
    elif np.argmax(lis)==7:
        save_model(df,model3,(1,2),20)

    elif np.argmax(lis)==8:
        save_model(df,model3,(1,3),30)

    #add all results to dataframe
    result=pd.concat([res1,res2,res3,res12,res22,res32,res13,res23,res33],axis=1)
    
    return result


result=main()





















