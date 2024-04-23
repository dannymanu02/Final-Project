# ABOUT

## Abstract or Overview:
This project aims to provide users with a comprehensive analysis of news articles from the subreddit r/worldnews, along with a sentiment prediction algorithm for news headlines. Through this web application, users can gain insights into global news trends and sentiments, empowering them to make informed decisions and stay updated on current events. The analysis offers valuable information on popular topics, sentiment trends, and key insights derived from the data. Additionally, the sentiment prediction algorithm allows users to predict the sentiment of news headlines, aiding in gauging public opinion and sentiment surrounding various news topics.

## Stakeholders:
The primary stakeholders who would benefit from our tool include journalists, policymakers, researchers, and individuals interested in global news and sentiment analysis. Journalists can use the tool to track news trends and sentiment surrounding specific topics, helping them identify important stories and gauge public reactions. Policymakers can leverage the insights to understand public opinion on relevant issues and tailor their policies accordingly. Researchers can utilize the data and algorithms for academic studies and analysis. Overall, our tool provides valuable insights for anyone interested in staying informed about global news trends and sentiments.

## Data Description:
We extracted data from the subreddit r/worldnews, which contains news articles and discussions on global events and topics. The dataset includes information such as article titles, publication dates, and user engagement metrics. Additionally, we merged this data with a news headline dataset from Kaggle, which provided additional headlines for sentiment prediction. The data underwent cleaning and preprocessing to remove duplicates, handle missing values, and ensure consistency for analysis and modeling.

## Algorithm Description:
The web application utilizes a sentiment prediction algorithm to predict the sentiment (positive, negative, or neutral) of news headlines. The algorithm is trained on a labeled dataset using machine learning techniques, such as natural language processing (NLP) and sentiment analysis. It analyzes the text of news headlines to classify them into different sentiment categories. The prediction results provide users with insights into the overall sentiment of news headlines, enabling them to understand public opinion and sentiment trends. The specific algorithm used in this case is called LSTM, LSTM stands for Long Short-Term Memory, and it's a type of artificial neural network architecture used in machine learning for processing and making predictions based on sequential data, like time series or text. 

### Here's how it works:
**Long-term Memory**: LSTM networks have a special ability to remember information from earlier in a sequence for a long time. This helps them capture important patterns and dependencies in the data that might span over many time steps.
**Short-term Memory**: At the same time, LSTM networks are also good at focusing on more recent information in the sequence. This allows them to adapt quickly to changes and updates in the data.
**Gate Mechanisms**: LSTMs achieve this by using special "gate" mechanisms that control the flow of information. These gates decide which information to keep, which to discard, and which to pass along to the next step in the sequence.

I have used a 3 layered neural network out of which the first layer is an embedding layer which converts input data, such as words or categorical variables, into dense, lower-dimensional vectors called embeddings. The second layer is an LSTM model. The third layer is the output layer which is a softmax layer for predictions.

## Tools Used:
- Streamlit: Used for building the web application and creating an interactive user interface.
- Python: Programming language used for data preprocessing, analysis, and modeling.
- Pandas: Library used for data manipulation and analysis.
- Scikit-learn: Library used for machine learning algorithms and sentiment analysis.
- NLTK (Natural Language Toolkit): Library used for natural language processing tasks such as tokenization and sentiment analysis.
- Tensorflow: Library for the sentiment prediction model
- Plotly: Library for plotting all the visualisations

## Ethical Concerns:
One ethical consideration is the potential bias in the data, as the news articles and headlines are sourced from online platforms and may reflect certain perspectives or biases. To mitigate this risk, we ensured transparency in our data collection and preprocessing methods, documenting any biases or limitations in the dataset. Additionally, we implemented measures to handle sensitive information responsibly and protect user privacy. It's important to interpret the results of sentiment analysis with caution, considering the limitations and potential biases in the data. Overall, we prioritize ethical considerations and strive to provide accurate and unbiased insights through our web application.
