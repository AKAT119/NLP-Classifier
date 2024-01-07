# NLP Classifier for YouTube Comments

Introduction

This project focuses on building a Natural Language Processing (NLP) classifier to distinguish between spam and non-spam (ham) comments on YouTube videos. It utilizes Python libraries like Pandas, NLTK, and scikit-learn to preprocess text data and apply machine learning for classification.
recognizing spam or ham comments on YouTubeInstallation

To run this project, follow these steps:

1- Clone the repository.
2- Install the required Python packages: pip install pandas numpy nltk sklearn plotly
3- Download the necessary NLTK data:
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')

Dataset

The dataset used is 'Youtube02-KatyPerry.csv', containing YouTube comments with labels for spam and ham.
The dataset is preprocessed for NLP tasks, including text cleaning, tokenization, removal of stopwords, stemming, and lemmatization.

Feature Extraction

The CountVectorizer from scikit-learn is used to convert text data into a matrix of token counts.
Then, Tf-idf Transformer is applied to reflect the importance of a term in the document and the entire corpus.


Model Training and Evaluation

A Multinomial Naive Bayes classifier is trained on the processed dataset.
The model's performance is evaluated using cross-validation and metrics like accuracy, confusion matrix, and classification report.
