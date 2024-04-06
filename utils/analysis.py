import plotly.graph_objects as go
import pandas as pd
import numpy as np
import plotly.express as px

import streamlit as st
class Analysis(object):

    @st.cache_data
    def pos_neg_news(df_reddit):
        top_articles = df_reddit[df_reddit['Post_Category'] == 'Top']

        top_articles_grouped = top_articles.groupby("Sentiment_Label").agg("Upvotes").sum().reset_index()
        top_articles_grouped.columns = ["Sentiment_Label", "Upvotes"]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=top_articles_grouped["Sentiment_Label"],
            y=top_articles_grouped["Upvotes"],
            marker=dict(color=['#FF6347', '#87CEEB', '#32CD32']), 
            hovertemplate='<b>%{x}</b><br>Count: %{y}', 
        ))

        fig.update_layout(
            title='Sentiment Distribution of Top Articles',
            xaxis=dict(title='Sentiment'),
            yaxis=dict(title='Count'),
            font=dict(family='Arial', size=12),
            plot_bgcolor='rgba(0, 0, 0, 0)', 
            hovermode='x',
        )

        return fig
    
    @st.cache_data
    def flair_cnt(df):
        flair_cnts = df["Flair"].value_counts().reset_index()
        flair_cnts.head()
        fig = px.bar(flair_cnts, x='Flair', y='count', title='Counts of Flairs', labels={'Flair': 'Flair', 'count': 'Count'})
        fig.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',
                        marker_line_width=1.5, opacity=0.6)
        fig.update_layout(xaxis_tickangle=-45)

        return fig
    
    @st.cache_data
    def upvote_distribution(df):
        fig = px.histogram(x=df["Upvote_Ratio"], nbins=10, histnorm='probability density',
                   title='Distribution of Upvote Ratios', labels={'x': 'Upvote Ratio', 'y': 'Density'})
        fig.update_layout(bargap=0.1)

        return fig
    
    @st.cache_data
    def word_counts_plot(df):
        fig = px.bar(df.head(10), x='Word', y='Frequency', color='Word',
             labels={'Counts': 'Counts'},
             title='Top 10 Most Frequent Words')
        fig.update_layout(xaxis_title='Words', yaxis_title='Counts')
        
        return fig
