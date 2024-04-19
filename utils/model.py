import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import numpy as np

class model(object):

    def predictor(model, string, maxlen, tokenizer):
        try:
            res = ""
            new_texts = [string]
            new_sequences = tokenizer.texts_to_sequences(new_texts)
            padded_new_sequences = pad_sequences(new_sequences, maxlen = maxlen, padding='post')
            predictions = model.predict(padded_new_sequences)
            sentiment_labels = ['Negative', 'Neutral', 'Positive']  # 0: Negative, 1: Neutral, 2: Positive

            for text, prediction in zip(new_texts, predictions):
                res = sentiment_labels[np.argmax(prediction)]
            return res
        except Exception as e:
            print(e)
    def token_generator(df_titles):
        try:
            tokenizer = Tokenizer()
            df_titles['Cleaned_Title'] = df_titles['Cleaned_Title'].astype(str)
            df_titles['Cleaned_Title'] = df_titles['Cleaned_Title'].str.lower().replace('[^\w\s]', '', regex=True)
            tokenizer.fit_on_texts(df_titles['Cleaned_Title'])
            sequences = tokenizer.texts_to_sequences(df_titles['Cleaned_Title'])
            maxlen = max(len(seq) for seq in sequences)

            return tokenizer, maxlen
        except Exception as e:
            print(e)
    
    def load_model_ltsm():
        try: 
            return tf.keras.models.load_model('LTSM.keras') 
        except Exception as e:
            print(e)