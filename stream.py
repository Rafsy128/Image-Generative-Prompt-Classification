import streamlit as st
from streamlit_option_menu import option_menu

# Fungsi untuk membaca file CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Memuat file CSS
local_css("styles.css")

# Navigasi Sidebar
with st.sidebar:
    selected = option_menu('Hitung Luas Bangun',
                           ['Training & Testing Model',
                            'Classify Prompts',
                            'Input a Prompt'],
                           default_index=0)

# Halaman Training & Testing Model
if selected == 'Training & Testing Model':

    try:
        import json
        import os
        
        import pandas as  pd
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
        
        
        from sklearn.preprocessing import FunctionTransformer
        from sklearn.base import BaseEstimator, TransformerMixin
        from sklearn.pipeline import FeatureUnion
        from sklearn.feature_extraction import DictVectorizer
        
        import swifter
        
        tqdm.pandas()
    except Exception as e:
        print("Error : {} ".format(e))

    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')

    import chardet

    # Detect encoding
    with open('dataset kel sam (training) 4.csv', 'rb') as f:
        result = chardet.detect(f.read())
        encoding = result['encoding']
        print(f"Detected encoding: {encoding}")

    # Read the CSV file with the detected encoding
    df = pd.read_csv('dataset kel sam (training) 4.csv', encoding=encoding, on_bad_lines='skip')
    st.write('Dataset: ')
    st.dataframe(df)

    # Print the first few rows to verify it loaded correctly
    st.dataframe(df.head())

    sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')

    st.write(df['kelas'].value_counts().plot( kind='bar', figsize=(15,10)))

    st.dataframe(df.columns)

    df.describe()

    df.isna().sum()

    df['kelas'].unique()

    df.head(2)

    df['kelas'].unique()

    df['deskripsi'] = df['deskripsi'].fillna('')

    import nltk
    nltk.download('stopwords')

    stop_words_ = set(stopwords.words('english'))
    wn = WordNetLemmatizer()

    def black_txt(token):
        return  token not in stop_words_ and token not in list(string.punctuation)  and len(token)>2
    
    def clean_txt(text):
        clean_text = []
        clean_text2 = []
        text = re.sub("'", "",text)
        text=re.sub("(\\d|\\W)+"," ",text)    
        clean_text = [ wn.lemmatize(word, pos="v") for word in word_tokenize(text.lower()) if black_txt(word)]
        clean_text2 = [word for word in clean_text if black_txt(word)]
        return " ".join(clean_text2)
    
    def subj_txt(text):
        return  TextBlob(text).sentiment[1]

    def polarity_txt(text):
        return TextBlob(text).sentiment[0]

    def len_text(text):
        if len(text.split())>0:
            return len(set(clean_txt(text).split()))/ len(text.split())
        else:
            return 0
        
    df['text'] = df['deskripsi']  

    df['text'] = df['text'].swifter.apply(clean_txt)
    df['polarity'] = df['text'].swifter.apply(polarity_txt)
    df['subjectivity'] = df['text'].swifter.apply(subj_txt)
    df['len'] = df['text'].swifter.apply(lambda x: len(x))

    X = df[['text', 'polarity', 'subjectivity','len']]
    y =df['kelas']

    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    v = dict(zip(list(y), df['kelas'].to_list()))

    text_clf = Pipeline([
    ...     ('vect', CountVectorizer(analyzer="word", stop_words="english")),
    ...     ('tfidf', TfidfTransformer(use_idf=True)),
    ...     ('clf', MultinomialNB(alpha=.01)),
    ... ])

    text_clf.fit(x_train['text'].to_list(), list(y_train))

    import numpy as np

    X_TEST = x_test['text'].to_list()
    Y_TEST = list(y_test)
    predicted = text_clf.predict(X_TEST)
    c = 0

    for doc, category in zip(X_TEST, predicted):
        
        if c == 2:break
        
        print("-"*55)
        print(doc)
        print(v[category])
        print("-"*55)

        c = c + 1 

    np.mean(predicted == Y_TEST)

    docs_new = ['a colorfoul and liquid 2D advertising background made of three streams of thick fruit juice in orange, light green and light blue colours']

    predicted = text_clf.predict(docs_new)

    v[predicted[0]]

    st.title('Generative Image Prompt Classification')

    # Input teks untuk prompt dengan placeholder
    prompt = st.text_input("Training & Testing a Model", placeholder="Write here ..")

    # Tombol untuk memproses prompt
    classify = st.button("Classify Prompt")

    if classify:
        # Fungsi dummy untuk klasifikasi prompt
        # Gantilah dengan model klasifikasi sesungguhnya
        def classify_prompt(prompt):
            # Proses klasifikasi di sini
            # Misal mengembalikan kategori dummy
            return "Category A"

        result = classify_prompt(prompt)
        st.write("Prompt Classification Result: ", result)
        st.success(f"Prompt '{prompt}' classified as: {result}")

# Halaman Classify Prompts
if selected == 'Classify Prompts':
    st.title('Generative Image Prompt Classification')

    # Input teks untuk prompt dengan placeholder
    prompt = st.text_input("Input a Prompt that you wish to Classify", placeholder="Write here ..")

    # Tombol untuk memproses prompt
    classify = st.button("Classify Prompt")

    if classify:
        # Fungsi dummy untuk klasifikasi prompt
        # Gantilah dengan model klasifikasi sesungguhnya
        def classify_prompt(prompt):
            # Proses klasifikasi di sini
            # Misal mengembalikan kategori dummy
            return "Category B"

        result = classify_prompt(prompt)
        st.write("Prompt Classification Result: ", result)
        st.success(f"Prompt '{prompt}' classified as: {result}")

# Halaman Input a Prompt
if selected == 'Input a Prompt':
    st.title('Generative Image Prompt Classification')

    # Input teks untuk prompt dengan placeholder
    prompt = st.text_input("Classify Your Full Dataset of Prompts", placeholder="Write here ..")

    # Tombol untuk memproses prompt
    classify = st.button("Classify Prompt")

    if classify:
        # Fungsi dummy untuk klasifikasi prompt
        # Gantilah dengan model klasifikasi sesungguhnya
        def classify_prompt(prompt):
            # Proses klasifikasi di sini
            # Misal mengembalikan kategori dummy
            return "Category C"

        result = classify_prompt(prompt)
        st.write("Prompt Classification Result: ", result)
        st.success(f"Prompt '{prompt}' classified as: {result}")
