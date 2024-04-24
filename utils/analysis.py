import plotly.graph_objects as go
import pandas as pd
import numpy as np
import plotly.express as px
from statsmodels.tsa.seasonal import seasonal_decompose

import streamlit as st

class Analysis(object):

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
    
    def flair_cnt(df):
        flair_cnts = df["Flair"].value_counts().reset_index()
        flair_cnts.head()
        fig = px.bar(flair_cnts, x='Flair', y='count', title='Counts of Flairs', labels={'Flair': 'Flair', 'count': 'Count'})
        fig.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',
                        marker_line_width=1.5, opacity=0.6)
        fig.update_layout(xaxis_tickangle=-45)

        return fig
    
    def upvote_distribution(df):
        fig = px.histogram(x=df["Upvote_Ratio"], nbins=10, histnorm='probability density',
                   title='Distribution of Upvote Ratios', labels={'x': 'Upvote Ratio', 'y': 'Density'})
        fig.update_layout(bargap=0.1)

        return fig
    
    def word_counts_plot(df):
        fig = px.bar(df.head(10), x='Word', y='Frequency', color='Word',
             labels={'Counts': 'Counts'},
             title='Top 10 Most Frequent Words')
        fig.update_layout(xaxis_title='Words', yaxis_title='Counts')
        
        return fig
    
    def corr_heatmap(df):
        df['Flair'] = df['Flair'].astype('category').cat.codes
        df['Post_Category'] = df['Post_Category'].astype('category').cat.codes
        correlation_matrix = df[['Num_Comments', 'Upvotes', 'Downvotes', 'Upvote_Ratio', 'Top_Comment_Score']].corr()

        heatmap = go.Heatmap(z=correlation_matrix.values,
                     x=correlation_matrix.columns,
                     y=correlation_matrix.columns,
                     colorscale='Viridis')

        layout = go.Layout(title='Correlation Matrix Heatmap',
                        xaxis=dict(title='Columns'),
                        yaxis=dict(title='Columns'))

        fig = go.Figure(data=[heatmap], layout=layout)

        return fig
    
    def timeseries_analysis(df):
        df['Date_Posted'] = pd.to_datetime(df['Date_Posted'])

        df.set_index('Date_Posted', inplace=True)

        numeric_columns = ['Num_Comments', 'Upvotes', 'Downvotes', 'Upvote_Ratio', 'Top_Comment_Score']
        resampled_data = df[numeric_columns].resample('M').mean()

        resampled_data.fillna(resampled_data.mean(), inplace=True)

        decomposition = seasonal_decompose(resampled_data['Upvotes'], model='additive')

        trend = go.Scatter(x=decomposition.trend.index, y=decomposition.trend, mode='lines', name='Trend')
        seasonal = go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal, mode='lines', name='Seasonal')
        residual = go.Scatter(x=decomposition.resid.index, y=decomposition.resid, mode='lines', name='Residual')

        layout = go.Layout(title='Seasonal Decomposition of Upvotes',
                        xaxis=dict(title='Date'),
                        yaxis=dict(title='Upvotes'))

        fig = go.Figure(data=[trend, seasonal, residual], layout=layout)
        
        return fig
