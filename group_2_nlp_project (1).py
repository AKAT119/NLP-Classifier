#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
# Using graph_objects
get_ipython().system('pip install plotly')
import plotly.graph_objects as go
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')


# In[2]:


# Data loading
filename = 'Youtube02-KatyPerry.csv'
data = pd.read_csv(filename)


# In[3]:


data.head(10)


# In[4]:


# Using graph_objects
fig = go.Figure([go.Scatter(x=data['DATE'])])
fig.show()


# In[5]:


print("Size of the data set is :",len(data) ,"&&","Number of unique values in COMMENT_ID column :",data["COMMENT_ID"].nunique())


# In[6]:


print("Size of the data set is :",len(data) ,"&&","Number of unique values in AUTHOR column :",data["AUTHOR"].nunique())


# In[7]:


print("Size of the data set is :",len(data) ,"&&","Number of unique values in DATE column :",data["DATE"].nunique())


# In[8]:


print("Size of the data set is :",len(data) ,"&&","Number of unique values in CONTENT column :",data["CONTENT"].nunique())


# In[9]:


print("Size of the data set is :",len(data) ,"&&","Number of unique values in CLASS column :",data["CLASS"].nunique())


# In[10]:


print(pd.unique(data["CLASS"]))


# In[11]:


print(data["CLASS"].value_counts())


# In[12]:


data = data.drop(columns=['COMMENT_ID', 'AUTHOR','DATE'])


# In[13]:


data


# In[14]:


data_X = data["CONTENT"]
data_X


# In[15]:


#library that contains punctuation
import string
string.punctuation

#defining the function to remove punctuation
def remove_punctuation(text):
    punctuationfree="".join([i for i in text if i not in string.punctuation])
    return punctuationfree
#storing the puntuation free text
data['CONTENT']= data['CONTENT'].apply(lambda x:remove_punctuation(x))


data['CONTENT']= data['CONTENT'].apply(lambda x: x.lower())




#defining function for tokenization
import re
def tokenization(text):
    tokens = re.split('W+',text)
    return tokens
#applying function to the column
data['CONTENT']= data['CONTENT'].apply(lambda x: tokenization(x))




#importing nlp library
import nltk
#Stop words present in the library
stopwords = nltk.corpus.stopwords.words('english')

#defining the function to remove stopwords from tokenized text
def remove_stopwords(text):
    output= [i for i in text if i not in stopwords]
    return output

#applying the function
data['CONTENT']= data['CONTENT'].apply(lambda x:remove_stopwords(x))



#importing the Stemming function from nltk library
from nltk.stem.porter import PorterStemmer
#defining the object for stemming
porter_stemmer = PorterStemmer()

#defining a function for stemming
def stemming(text):
    stem_text = [porter_stemmer.stem(word) for word in text]
    return stem_text
data['CONTENT']=data['CONTENT'].apply(lambda x: stemming(x))




from nltk.stem import WordNetLemmatizer
#defining the object for Lemmatization
wordnet_lemmatizer = WordNetLemmatizer()

#defining the function for lemmatization
def lemmatizer(text):
    lemm_text = [wordnet_lemmatizer.lemmatize(word) for word in text]
    return lemm_text
data['CONTENT']=data['CONTENT'].apply(lambda x:lemmatizer(x))


# In[16]:


data['CONTENT'] = data['CONTENT'].astype(str)


# In[17]:


print(data["CONTENT"])


# In[18]:


data


# In[19]:


new_data = data.sample(frac=1)


# In[20]:


new_data


# In[21]:


train=new_data.sample(frac=0.75,random_state=10)
test=new_data.drop(train.index)


# In[22]:


train


# In[23]:


train["CONTENT"]


# In[24]:


train["CLASS"]


# In[25]:


# Build a count vectorizer and extract term counts 
#Text preprocessing, tokenizing and filtering of stopwords are all included in
#CountVectorizer, which builds a dictionary of features and transforms documents
# to feature vectors:
count_vectorizer = CountVectorizer()
train_x = count_vectorizer.fit_transform(train["CONTENT"])
print("\nDimensions of training data:", train_x.shape)


# In[26]:


#This downscaling is called tf–idf for “Term Frequency times Inverse Document Frequency”.
# Create the tf-idf transformer
tfidf = TfidfTransformer()
train_tfidf = tfidf.fit_transform(train_x)
type(train_tfidf)


# In[27]:


print(train_tfidf)


# In[28]:


train_tfidf


# In[29]:


from sklearn.naive_bayes import MultinomialNB


# In[30]:


# Train a Multinomial Naive Bayes classifier
classifier = MultinomialNB().fit(train_tfidf, train["CLASS"])


# In[31]:


from sklearn.model_selection import cross_val_score


# In[32]:


#score of the accuracy generated with the training data
scores = cross_val_score(classifier,train_tfidf,train["CLASS"],cv=5) 
print(scores)


# In[33]:


scores.mean()*100


# In[34]:


test_x = test["CONTENT"]
test_x
test_y = test["CLASS"]


# In[35]:


# Transform the testing feature data using count vectorizer
test_tc = count_vectorizer.transform(test_x)
type(test_tc)


# In[36]:


# Transform vectorized data using tfidf transformer
test_tfidf = tfidf.transform(test_tc) 
type(test_tfidf)


# In[37]:


# Predict the output categories, fitting the transformed testing feature
y_pred = classifier.predict(test_tfidf) 

#Accuracy generated with the testing data
print(accuracy_score(test_y, y_pred)) 
print(confusion_matrix(test_y, y_pred)) 
print(classification_report(test_y, y_pred))

