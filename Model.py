# import tweepy
import sys
import csv
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report 
from sklearn.metrics import auc
from sklearn.metrics import roc_curve

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split



import regex as re
import pickle
#import the dataset
df1=pd.read_csv("D:\Techmiya\PRoject\Datasets\hatespeech\labeled_data.csv")
df2=pd.read_csv("HOT_preprocessed_data.csv")  
#Doing some adjustments

c=df1['class']
df1.rename(columns={'tweet' : 'text',
                   'class' : 'category'}, 
                    inplace=True)
a=df1['text']
b=df1['category'].map({0: 'hate_speech', 1: 'offensive_language',2: 'neither'})

df= pd.concat([a,b,c], axis=1)
df.rename(columns={'class' : 'label'}, 
                    inplace=True)
df
#Replacing the values to ease understanding. (Assigning 1 to Positive sentiment 4)
df2['score'] = df2['score'].replace([0,1,2],[2,0,1])
df=df.drop(['category'],axis=1)
df2=df2.drop(['old_text'],axis=1)
df2.rename(columns = {'score':'label'}, inplace = True)
data = pd.concat([df,df2]) 
#Creating a copy 
clean_reviews=data.copy()
def review_cleaning(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text
data['text']=data['text'].apply(lambda x:review_cleaning(x))
data.head()
stop_words= ['yourselves', 'between', 'whom', 'itself', 'is', "she's", 'up', 'herself', 'here', 'your', 'each', 
             'we', 'he', 'my', "you've", 'having', 'in', 'both', 'for', 'themselves', 'are', 'them', 'other',
             'and', 'an', 'during', 'their', 'can', 'yourself', 'she', 'until', 'so', 'these', 'ours', 'above', 
             'what', 'while', 'have', 're', 'more', 'only', "needn't", 'when', 'just', 'that', 'were', "don't", 
             'very', 'should', 'any', 'y', 'isn', 'who',  'a', 'they', 'to', 'too', "should've", 'has', 'before',
             'into', 'yours', "it's", 'do', 'against', 'on',  'now', 'her', 've', 'd', 'by', 'am', 'from', 
             'about', 'further', "that'll", "you'd", 'you', 'as', 'how', 'been', 'the', 'or', 'doing', 'such',
             'his', 'himself', 'ourselves',  'was', 'through', 'out', 'below', 'own', 'myself', 'theirs', 
             'me', 'why', 'once',  'him', 'than', 'be', 'most', "you'll", 'same', 'some', 'with', 'few', 'it',
             'at', 'after', 'its', 'which', 'there','our', 'this', 'hers', 'being', 'did', 'of', 'had', 'under',
             'over','again', 'where', 'those', 'then', "you're", 'i', 'because', 'does', 'all']    
data['text'] = data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
#Separating input feature and label
X=data.text
y=data.label

from sklearn.feature_extraction.text import CountVectorizer
# Create word vector (count)
CountVector = CountVectorizer(max_features=2000)

X = CountVector.fit_transform(data.text).toarray()
y = data.label.values

print(X.shape, y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=555)

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC

rfClf = RandomForestClassifier(n_estimators=500, random_state=0) # 500 trees.
svmClf = SVC(probability=True, random_state=0) # probability calculation


# constructing the ensemble classifier by mentioning the individual classifiers.
clf2 = VotingClassifier(estimators = [('rf',rfClf), ('svm',svmClf)], voting='soft') 

# train the ensemble classifier
clf2.fit(X_train, y_train) 
from sklearn.metrics import precision_score, accuracy_score
y_actual, y_pred = y_test, clf2.predict(X_test)

accuracy_score_VC_test = accuracy_score(y_actual, y_pred)

print('The accuracy score of Voting classifier on Test is : ',round(accuracy_score_VC_test * 100,2), '%')

pickle.dump(clf2, open('model.pkl','wb'))



