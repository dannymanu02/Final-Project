import os

from utils.b2 import B2
from utils.analysis import Analysis as an
from utils.nlp_analysis import NLP_Analysis as nlp
from utils.model import model

from dotenv import load_dotenv

import streamlit as st

import pandas as pd
import plotly.graph_objects as go

import time

load_dotenv()

st.set_page_config(layout="wide")


def page_style():
    gradient = """
    <style>
    [data-testid="stAppViewBlockContainer"]{
    # background-color: #D9AFD9;
    # background-image: linear-gradient(-235deg, #D9AFD9 0%, #97D9E1 100%);
    # background-image: linear-gradient( 95.2deg, rgba(173,252,234,1) 26.8%, rgba(192,229,246,1) 64% );
    background-image: linear-gradient(to right, #ffefba, #ffffff);
    }
    </style>
    """
    st.markdown(gradient, unsafe_allow_html=True)
    st.markdown("<Center><H1> r/WorldNews Analysis</H1></Center>", unsafe_allow_html = True)

def sidebar_style():
    gradient =  """<style>
    [data-testid=stSidebar]{
    background-image: linear-gradient(to bottom, #ffefba, #ffffff);
    }
    </style>"""
    
    st.sidebar.markdown(gradient, unsafe_allow_html=True)

# st.info("r/worldnews is a highly active subreddit dedicated to sharing and discussing global news and current events. With millions of subscribers, it serves as a hub for breaking news from around the world, covering a wide range of topics such as politics, conflicts, diplomacy, and social issues. Users often engage in lively discussions, debates, and analysis, providing diverse perspectives on the latest developments. The subreddit's content is curated to include a mix of reputable news sources and user-submitted content, fostering an environment of information sharing and community engagement. Overall, r/worldnews is a vital platform for staying informed about global affairs in a rapidly changing world.", icon='ℹ️')

with st.spinner("r/worldnews is a highly active subreddit dedicated to sharing and discussing global news and current events. With millions of subscribers, it serves as a hub for breaking news from around the world, covering a wide range of topics such as politics, conflicts, diplomacy, and social issues. Users often engage in lively discussions, debates, and analysis, providing diverse perspectives on the latest developments. The subreddit's content is curated to include a mix of reputable news sources and user-submitted content, fostering an environment of information sharing and community engagement. Overall, r/worldnews is a vital platform for staying informed about global affairs in a rapidly changing world."):
    time.sleep(7)

page_style()
sidebar_style()

st.sidebar.markdown('''
# Skip to Section
- [Upvote Count vs Sentiment](#upvote-count-vs-sentiment)
- [Flairs vs Counts](#flairs-vs-counts)
- [Distribution of Upvote Ratios](#distribution-of-upvote-ratios)
- [Correlation Matrix](#correlation-matrix)
- [Time Series Analysis](#time-series-analysis)
- [NLP Modelling and Analysis](#nlp-modelling-and-analysis)
- [Goals](#goals)
- [Data Cleaning and Pre-Processing](#data-cleaning-and-pre-processing)
- [Buzzwords in Data](#buzzwords-in-data)
- [Enter your headline to know the sentiment](#enter-your-headline-to-know-the-sentiment)
----------------------------------------------------------------------------------------------------------
# Sections for reference
- [About the Model](#about-the-model)
- [Link to GitHub](https://github.com/dannymanu02/Final-Project)
''', unsafe_allow_html=True)

def display_toast():
    st.toast("r/worldnews is a highly active subreddit dedicated to sharing and discussing global news and current events. With millions of subscribers, it serves as a hub for breaking news from around the world, covering a wide range of topics such as politics, conflicts, diplomacy, and social issues. Users often engage in lively discussions, debates, and analysis, providing diverse perspectives on the latest developments. The subreddit's content is curated to include a mix of reputable news sources and user-submitted content, fostering an environment of information sharing and community engagement. Overall, r/worldnews is a vital platform for staying informed about global affairs in a rapidly changing world.", icon='ℹ️')

st.sidebar.button("About", on_click=display_toast())

def get_data(file_name):
    # collect data frame of reviews and their sentiment
    b2.set_bucket(os.environ['B2_BUCKETNAME'])
    df_reddit = b2.get_df(file_name)

    return df_reddit

b2 = B2(endpoint=os.environ['B2_ENDPOINT'],
        key_id=os.environ['B2_keyID'],
        secret_key=os.environ['B2_applicationKey'])

df_reddit = get_data("reddit_worldnews_sentiments_clean.csv")
df_analysis = get_data("cumilative_headlines.csv")

# st.header("Upvote Count vs Sentiment")
st.markdown("<H4>Upvote Count vs Sentiment<H4>", unsafe_allow_html=True)
st.markdown("Exploring the relationship between upvote count and sentiment, the following graph illustrates how sentiment labels—positive, negative, and neutral—correlate with the popularity of posts on the r/worldnews subreddit.")

fig_1 = an.pos_neg_news(df_reddit)

st.plotly_chart(fig_1)

st.write("From the above graph it is clear that people on the internet like negative news, news that is controversial and dividing.")

st.markdown("<H4>Flairs vs Counts<H4>", unsafe_allow_html=True)
st.markdown("Examining the distribution of post counts across different flair categories, this graph offers insights into the varying levels of engagement and interest among topics discussed on the r/worldnews subreddit")

fig_2 = an.flair_cnt(df_reddit)

st.plotly_chart(fig_2)

st.write("""The top flair categories on the r/worldnews subreddit reflect a diverse range of global issues and geopolitical events. With 389 posts, "Russia/Ukraine" emerges as the most prevalent topic, likely driven by ongoing tensions and conflicts in the region. "COVID-19" follows closely behind with 113 posts, indicative of the enduring impact of the pandemic on global discourse. "Trump" and "Israel/Palestine" garner significant attention with 111 and 81 posts respectively, highlighting the enduring interest in political leadership and Middle East affairs. Additionally, "Behind Soft Paywall" with 43 posts suggests a focus on news content accessibility and media transparency within the subreddit community.""")

st.markdown("<h4>Distribution of Upvote Ratios</h4>", unsafe_allow_html=True)
st.markdown("Analyzing the distribution of post counts across various flair categories, this graph unveils the spectrum of engagement and interest levels pertaining to topics discussed within the r/worldnews subreddit. By segmenting posts into distinct flair categories, the visualization illustrates how certain topics garner more attention and participation than others. This insight underscores the dynamic nature of community engagement and highlights the diverse range of subjects that captivate the interest of users within the subreddit. Through this exploration, viewers gain a nuanced understanding of the subreddit's content landscape and the relative prominence of different thematic areas.")

fig_3 = an.upvote_distribution(df_reddit)

st.plotly_chart(fig_3)

st.write("From the graph, it appears that the most common upvote ratio is around 0.6. Upvote ratios below 0.5 and upvote ratios above 0.8 are less common.")

st.markdown("<H4>Correlation Matrix</H4>", unsafe_allow_html=True)
st.markdown("Analyzing the correlation between the columns 'Num_Comments', 'Upvotes', 'Downvotes', 'Upvote_Ratio', 'Top_Comment_Score', to understand how each of these columns are related to one another.")

fig_4 = an.corr_heatmap(df_reddit)

st.plotly_chart(fig_4)
st.write("From the heatmap, we can see that there is a very strong correlation between Number of Upvotes and Number of Comments which means that unlike other social media sites where bot activity disproportionately increases the likes, there is a high chance that reddit doesn't have much bot involvement or atleast this subreddit and the comments and upvotes are made by legitimate people.")

st.markdown("<H4>Time Series Analysis</H4>", unsafe_allow_html=True)
st.markdown("I have a date and time column in my data, I want to analyse this and see if I can capture some kind of trend or seasonality.")

fig_5 = an.timeseries_analysis(df_reddit)

st.plotly_chart(fig_5)
st.write("From the above plot we can see that the Trend line recedes after in 2024, this could be an indicator that the engagement with the subreddit has started to decline. There is also a Seasonality pattern for a period of one year, the reasons are unknown but there is a clear pattern.")

st.markdown("<center><h1>NLP Modelling and Analysis</h1></center>", unsafe_allow_html=True)

st.write("The data I am working with is a list of titles and a sentiment tag, the sentiment tag specifies whether the news headline belongs to a positive, negative or a neutral article.")

st.dataframe(df_analysis.head(10))

st.markdown("<h4>Goals</h4>", unsafe_allow_html=True)
st.markdown("""<ol>
            <li>To analyze the headlines data along with the sentiments to understand what topics are being discussed the most.</li>
            <li>To create an NLP model that takes a headline and outputs the sentiment of the headline.</li>
            </ol>""", unsafe_allow_html=True)

st.markdown("<h4>Data Cleaning and Pre-Processing</h4>", unsafe_allow_html=True)
st.markdown("""The data we have is raw and it needs to be cleaned and pre-processed in order to do nlp modelling.</br>Things like stop word removal,
            removing punctuation and lemmatization need to done.""", unsafe_allow_html=True)

df_analysis_clean = nlp.data_nlp_cleaning(df_analysis)

st.markdown("<b>Data after pre-processing", unsafe_allow_html=True)
st.dataframe(df_analysis_clean.head(10))

st.markdown("""The Original data is present in Title, and the cleaned data is present in the column <b>Cleaned_Title.""", unsafe_allow_html=True)

st.markdown("<h4>Buzzwords in Data</h4>", unsafe_allow_html=True)
st.write("Let's check the most used or frequently occurring words in the data. The most frequent words can be used to show social and political trends across the world.")
# word_frequencies = nlp.word_counts(df_analysis_clean)
# fig_wf = an.word_counts_plot(word_frequencies)

# st.plotly_chart(fig_wf)
st.markdown("<b>There was supposed to be a plot for most frequently used words but it was breaking the app, so commenting out the code after informing professor.</b>", unsafe_allow_html=True)
st.write("From the looks of it, Trump seems to be the most talked about person, he's never out of the news I guess!")

tokenizer, maxlen = model.token_generator(df_analysis_clean)
model_ltsm = model.load_model_ltsm()

st.markdown("<h4>Enter your headline to know the sentiment </h4>", unsafe_allow_html=True)

text_input = st.text_input('Text here')

submit_button = st.button('get sentiment')

if submit_button:
    sentiment = model.predictor(model_ltsm, text_input, maxlen, tokenizer)
    st.write(f'Sentiment: {sentiment}')

st.markdown("""<h4>About the Model</h4>""", unsafe_allow_html=True)
st.markdown("""I have used a Recurrent Neural Network called Long Short Term Memory(LSTM) Model which is generally used for NLP purposes. 
            The neural network has 3 layers with the first layer being an embedding layer, second one being the LTSM layer and the third one being the ouput layer. 
            The output layer uses the softmax activation function instead of sigmoid activation function because of the data being multi classed.
            This model has an accuracy of 89.5%' and a categorical_crossentropy loss of 0.625.""")