import json
import os
import pandas as pd
import spacy
import seaborn as sns
import string
from tqdm import tqdm
from textblob import TextBlob
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import swifter
import numpy as np
import chardet

tqdm.pandas()

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Detect encoding of the CSV file
with open('dataset kel sam (training) 4.csv', 'rb') as f:
    result = chardet.detect(f.read())
    encoding = result['encoding']
    print(f"Detected encoding: {encoding}")

# Read the CSV file with the detected encoding
df = pd.read_csv('dataset kel sam (training) 4.csv', encoding=encoding, on_bad_lines='skip')
print('Dataset:')
print(df.head())

# Visualize missing values with a heatmap
sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')

# Plot the distribution of classes
df['kelas'].value_counts().plot(kind='bar', figsize=(15, 10))

print(df.columns)
print(df.describe())
print(df.isna().sum())
print(df['kelas'].unique())
print(df.head(2))
print(df['kelas'].unique())

# Fill missing descriptions with an empty string
df['deskripsi'] = df['deskripsi'].fillna('')

# Define stop words and lemmatizer
stop_words_ = set(stopwords.words('english'))
wn = WordNetLemmatizer()

# Function to filter tokens
def black_txt(token):
    return token not in stop_words_ and token not in list(string.punctuation) and len(token) > 2

# Function to clean text
def clean_txt(text):
    text = re.sub("'", "", text)
    text = re.sub("(\\d|\\W)+", " ", text)
    clean_text = [wn.lemmatize(word, pos="v") for word in word_tokenize(text.lower()) if black_txt(word)]
    return " ".join([word for word in clean_text if black_txt(word)])

# Functions to calculate sentiment and text length
def subj_txt(text):
    return TextBlob(text).sentiment[1]

def polarity_txt(text):
    return TextBlob(text).sentiment[0]

def len_text(text):
    if len(text.split()) > 0:
        return len(set(clean_txt(text).split())) / len(text.split())
    else:
        return 0

# Clean the text data
df['text'] = df['deskripsi']
df['text'] = df['text'].swifter.apply(clean_txt)
df['polarity'] = df['text'].swifter.apply(polarity_txt)
df['subjectivity'] = df['text'].swifter.apply(subj_txt)
df['len'] = df['text'].swifter.apply(lambda x: len(x))

# Prepare features and target variable
X = df[['text', 'polarity', 'subjectivity', 'len']]
y = df['kelas']

# Encode target variable
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Split data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Create a dictionary to map encoded labels back to original labels
v = dict(zip(list(y), df['kelas'].to_list()))

# Define and train the model pipeline
text_clf = Pipeline([
    ('vect', CountVectorizer(analyzer="word", stop_words="english")),
    ('tfidf', TfidfTransformer(use_idf=True)),
    ('clf', MultinomialNB(alpha=.01)),
])

text_clf.fit(x_train['text'].to_list(), list(y_train))

# Evaluate the model
X_TEST = x_test['text'].to_list()
Y_TEST = list(y_test)
predicted = text_clf.predict(X_TEST)

# Print first few predictions
c = 0
for doc, category in zip(X_TEST, predicted):
    if c == 2:
        break
    print("-" * 55)
    print(doc)
    print(v[category])
    print("-" * 55)
    c = c + 1

# Calculate accuracy
accuracy = np.mean(predicted == Y_TEST)
print(f"Accuracy: {accuracy}")

# Predict new data from user input
new_doc = input("Enter a new document for classification: ")
cleaned_new_doc = clean_txt(new_doc)
predicted = text_clf.predict([cleaned_new_doc])
print(f"Predicted class: {v[predicted[0]]}")
