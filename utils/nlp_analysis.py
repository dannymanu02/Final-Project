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
    
    def word_counts(df_titles):
        def chunk_generator(df, chunk_size=1000):
            try: 
                st.write("Before chunks")
                start = 0
                while start < len(df):
                    yield df_titles['Cleaned_Title'].iloc[start:start+chunk_size].tolist()  # Convert to list
                    start += chunk_size
                    st.write("Chunk: "+str(len(df)-start))
                st.write("After while")
            except Exception as e:
                print(e)
        try:
            vectorizer = CountVectorizer()

            fitted = False

            X = None

            for chunk in chunk_generator(df_titles):
                if not fitted:
                    vectorizer.fit(chunk)
                    fitted = True
                
                X_chunk = vectorizer.transform(chunk)
                if X is None:
                    X = X_chunk
                else:
                    features = vectorizer.get_feature_names_out()
                    
                    vectorizer.vocabulary_ = {feature: idx for idx, feature in enumerate(features)}
                    
                    X = scipy.sparse.vstack([X, X_chunk])

            word_counts = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
            word_frequencies = word_counts.sum().sort_values(ascending=False)
            word_frequencies.columns = ["word", "count"]
            word_frequencies_df = word_frequencies.reset_index()
            word_frequencies_df.columns = ['Word', 'Frequency']

            return word_frequencies_df
        except Exception as e:
            print(e)