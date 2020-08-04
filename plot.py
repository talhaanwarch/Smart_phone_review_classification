# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 08:10:54 2020

@author: TAC
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
import numpy as np
train=pd.read_csv('user_reviews.csv')
train_labels=train['score'].astype('int')

train['extract']=train['extract'].str.lower()
train['extract']=train['extract'].apply(lambda x:x.translate(str.maketrans('', '', string.punctuation)))
train_labels[train_labels<8]=0
train_labels[train_labels>=8]=1


def count_values_in_column(train_labels):
    total=train_labels.value_counts(dropna=False)
    percentage=round(train_labels.value_counts(dropna=False,normalize=True)*100,2)
    return pd.concat([total,percentage],axis=1,keys=['Total','Percentage'])

count_values_in_column(train_labels)

#plot countplot
sns.countplot(x='score',data=train_labels.reset_index())
plt.xticks(ticks=range(2),labels=['Not recommended','Recommended'])

# plt word length distribution
train['words_len']=train['extract'].apply(lambda x: len(x))
neg=train['words_len'][train_labels==0]
pos=train['words_len'][train_labels==1]

sns.kdeplot(pos,label='Recommended') 
sns.kdeplot(neg,label='Not Recommended')


import nltk
from nltk.corpus import stopwords
stop_words = stopwords.words("english") 
#stop_words.extend(['I','The'])
def remove_stopword(x):
    return [y for y in x if y not in stop_words]


from collections import Counter
train['temp_list'] = train['extract'].apply(lambda x:str(x).split())
train['temp_list'] =train['temp_list'].apply(lambda x:remove_stopword(x))
top = Counter([item for sublist in train['temp_list'] for item in sublist])
temp = pd.DataFrame(top.most_common(10))
temp.columns = ['Common_words','count']
temp.style.background_gradient(cmap='coolwarm')


temp['norm_count']=temp['count'].divide((temp['count'][0])) #no of all tweets
temp.style.background_gradient(cmap='coolwarm')



top = Counter([item for sublist in train[train_labels==1]['temp_list'] for item in sublist])
temp = pd.DataFrame(top.most_common(10))
temp.columns = ['Common_words','count']

temp['norm_count']=temp['count'].divide((temp['count'][0])) #no of all tweets
temp.style.background_gradient(cmap='coolwarm')











