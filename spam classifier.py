# -*- coding: utf-8 -*-
"""
Created on Sun feb 22 21:26:42 2022

@author: AJAY NALLA
"""

# -*- coding: utf-8 -*-
"""
Created on Sun feb 22 09:03:07 2022

@author: AJAY NALLA
"""

import pandas as pd
import os

data=pd.read_table(r"C:/Users/AJAY NALLA/OneDrive/Desktop/ml project/smsspamcollection/SMSSpamCollection",sep='\t',
                  names=("label","message")) 
#cleaning and preprcessing of data
import re
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
ps=PorterStemmer()
final=[]
for i in range( 0, len(data)):
    temp=re.sub('[^a-zA-z]',' ',data['message'][i])
    temp=temp.lower()
    temp=temp.split()
    
    temp=[ps.stem(word) for word in temp if not word in stopwords.words('english')]
    temp=' '.join(temp)
    final.append(temp)

#creating the bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000)
X=cv.fit_transform(final).toarray()
y=pd.get_dummies(data['label'])
y=y.iloc[:,1].values

#train test split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)
#training the model
from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train,y_train)
y_pred=spam_detect_model.predict(X_test)

from sklearn.metrics import confusion_matrix
confusion_m=confusion_matrix(y_test, y_pred)
accuracy=accuracy_score(y_test,y_pred)