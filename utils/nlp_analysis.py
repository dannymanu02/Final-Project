import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string

from sklearn.feature_extraction.text import CountVectorizer
import scipy.sparse

import pandas as pd

import streamlit as st

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class NLP_Analysis(object):

    @st.cache_data
    def data_nlp_cleaning(df_titles):
        stop_words = set(stopwords.words('english'))
        punctuations = set(string.punctuation)

        lemmatizer = WordNetLemmatizer()

        def preprocess_text(text):

            if pd.isna(text):
                return ""

            tokens = word_tokenize(text.lower())

            clean_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words and token not in punctuations]

            cleaned_text = ' '.join(clean_tokens)
            
            return cleaned_text

        df_titles['Cleaned_Title'] = df_titles['Title'].apply(preprocess_text)

        return df_titles
    
    @st.cache_data
    def word_counts(df_titles):
        def chunk_generator(df, chunk_size=1000):
            start = 0
            while start < len(df):
                yield df_titles['Cleaned_Title'].iloc[start:start+chunk_size].tolist()  # Convert to list
                start += chunk_size

        # Initialize CountVectorizer without limiting vocabulary size
        vectorizer = CountVectorizer()

        # Flag to indicate if the vectorizer is fitted
        fitted = False

        # Initialize an empty matrix to hold the combined data
        X = None

        # Process data in chunks and transform CountVectorizer
        for chunk in chunk_generator(df_titles):
            if not fitted:
                # Fit the vectorizer with the vocabulary from the first chunk
                vectorizer.fit(chunk)
                fitted = True
            
            X_chunk = vectorizer.transform(chunk)
            if X is None:
                X = X_chunk
            else:
                # Get the features in the current chunk
                features = vectorizer.get_feature_names_out()
                
                # Update the vocabulary with new features from the current chunk
                vectorizer.vocabulary_ = {feature: idx for idx, feature in enumerate(features)}
                
                # Vertically stack the sparse matrices
                X = scipy.sparse.vstack([X, X_chunk])

        # Convert sparse matrix to DataFrame and calculate word frequencies
        word_counts = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
        word_frequencies = word_counts.sum().sort_values(ascending=False)
        word_frequencies.columns = ["word", "count"]
        word_frequencies_df = word_frequencies.reset_index()
        word_frequencies_df.columns = ['Word', 'Frequency']

        return word_frequencies_df