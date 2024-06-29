import json
import os
import pandas as pd
import spacy
import seaborn as sns
import string
import matplotlib.pyplot as plt
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
import streamlit as st

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Set up Streamlit
# st.title('Text Classification App')

# File uploader for training dataset
uploaded_file = st.file_uploader("Upload your training CSV file", type=["csv"])

if uploaded_file is not None:
    # Detect encoding of the uploaded CSV file
    result = chardet.detect(uploaded_file.read())
    encoding = result['encoding']
    st.write(f"Detected encoding: {encoding}")

    # Read the CSV file with the detected encoding
    uploaded_file.seek(0)  # Reset file pointer to the beginning
    df = pd.read_csv(uploaded_file, encoding=encoding, on_bad_lines='skip')
    st.write('Training Dataset:')
    st.dataframe(df)

    # Visualize missing values with a heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
   # st.pyplot(plt.gcf())  # Pass the current figure to st.pyplot()

    # Plot the distribution of classes
    plt.figure(figsize=(15, 10))
    df['kelas'].value_counts().plot(kind='bar')
    st.pyplot(plt.gcf())  # Pass the current figure to st.pyplot()

    st.dataframe(df.columns)
    st.write(df.describe())
    st.write(df.isna().sum())
    st.write(df['kelas'].unique())
    st.write(df.head(2))
    st.write(df['kelas'].unique())

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
        st.write("-" * 55)
        st.write(doc)
        st.write(v[category])
        st.write("-" * 55)
        c = c + 1

    # Calculate accuracy
    accuracy = np.mean(predicted == Y_TEST)
    st.write(f"Accuracy: {accuracy}")

    # File uploader for prediction dataset
    st.subheader("Classify Your Full Dataset of Prompts")
    prediction_file = st.file_uploader("Upload Your Testing Dataset", type=["csv"])

    if prediction_file is not None:
        # Detect encoding of the uploaded CSV file
        result = chardet.detect(prediction_file.read())
        encoding = result['encoding']
        st.write(f"Detected encoding: {encoding}")

        # Read the CSV file with the detected encoding
        prediction_file.seek(0)  # Reset file pointer to the beginning
        df_pred = pd.read_csv(prediction_file, encoding=encoding, on_bad_lines='skip')
        st.write('Prediction Dataset:')
        st.dataframe(df_pred)

        # Fill missing descriptions with an empty string
        df_pred['deskripsi'] = df_pred['deskripsi'].fillna('')

        # Clean the text data in the prediction dataset
        df_pred['text'] = df_pred['deskripsi']
        df_pred['text'] = df_pred['text'].swifter.apply(clean_txt)

        # Predict classes for the new dataset
        predicted_classes = text_clf.predict(df_pred['text'].to_list())

        # Map encoded labels back to original labels
        df_pred['predicted_kelas'] = [v[label] for label in predicted_classes]

        st.write('Prediction Results:')
        st.dataframe(df_pred[['deskripsi', 'predicted_kelas']])
else:
    st.write("Please upload a training CSV file to get started.")
