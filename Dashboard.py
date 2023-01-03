import streamlit as st
import plotly.graph_objects as go
import snscrape.modules.twitter as sntwitter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import re
from collections import Counter


#---MSQL CONNECTION---#
import mysql.connector
from sqlalchemy import create_engine
conn_clean = create_engine('mysql+mysqlconnector://root:sukses25@localhost:3306/clean_tweets')
conn_date = create_engine('mysql+mysqlconnector://root:sukses25@localhost:3306/update_date')
conn_raw = create_engine('mysql+mysqlconnector://root:sukses25@localhost:3306/raw_tweets')

clean_query = "SELECT * FROM clean_tweets"
df_clean = pd.read_sql(clean_query, conn_clean)
df_clean.to_csv('clean_tweets.csv')
df_clean['clean_tweets'] = df_clean['clean_tweets'].replace('', np.nan)
df_clean = df_clean.dropna(subset=['clean_tweets'])
df_clean = df_clean.drop_duplicates(subset=['tweet_id'])
df_clean.to_csv('clean_tweets.csv')

from nltk.tokenize import word_tokenize

def word_tokenize_wrapper(text):
  return word_tokenize(text)

df_clean['lda_token'] = df_clean['clean_tweets'].apply(word_tokenize_wrapper)


date_query = "SELECT * FROM update_date ORDER BY last_update DESC LIMIT 1"
df_last_date = pd.read_sql(date_query, conn_date)
date_query_new = "SELECT * FROM update_date ORDER BY last_update LIMIT 1"
df_new_date = pd.read_sql(date_query_new, conn_date)

df_clean['lda_token'] = df_clean['clean_tweets'].apply(word_tokenize_wrapper)
negative_data = df_clean.loc[df_clean['label'] == "negative"]
positive_data = df_clean.loc[df_clean['label'] == "positive"]
neutral_data = df_clean.loc[df_clean['label'] == "neutral"]

negative_word = negative_data['clean_tweets'].astype(str)
negative_word = ' '.join(negative_word.tolist())

positive_word = positive_data['clean_tweets'].astype(str)
positive_word = ' '.join(positive_word.tolist())

#---END OF CONNECTION---#   

#---CONFIG---#
page_title = 'Recession Sentiment Dashboard'
page_icon = ":pushpin:" 
st.set_page_config(page_title=page_title, page_icon=page_icon, layout='wide')
#---END OF CONFIG---#

st.sidebar.header("KELOMPOK 9")
#------Barplot------#
st.title("Common Word :thought_balloon:")
st.markdown("#####")
bp_selection = ['Negative Sentiment', 'Positive Sentiment']

bp_result = st.selectbox("Select your data", bp_selection, key = '2')

if bp_result == 'Negative Sentiment':
    #Tokenizing
    #import word_tokenize dari nltk
    import nltk
    from nltk.tokenize import word_tokenize
    tokens = nltk.tokenize.word_tokenize(negative_word)
    from nltk.tokenize import word_tokenize
    from nltk.probability import FreqDist

    kemunculan = nltk.FreqDist(tokens)
    kemunculan2 = dict((k, v) for k, v in kemunculan.items() if v >= 5)
    # creating the dataset
    courses = list(kemunculan2.keys())
    values = list(kemunculan2.values())

    from collections import Counter
    negative_data['temp_list'] = negative_data['clean_tweets'].apply(lambda x:str(x).split())
    top = Counter([item for sublist in negative_data['temp_list'] for item in sublist])
    temp = pd.DataFrame(top.most_common(20))
    temp.columns = ['Common_words','count']
    temp.style.background_gradient(cmap='Blues')
    import plotly.express as px
    fig = px.bar(temp, x="count", y="Common_words", title='Commmon Words in negative sentiment', orientation='h', 
                width=1000, height=700,color='Common_words')
    st.plotly_chart(fig,height=800)
elif bp_result == 'Positive Sentiment':
    import nltk
    from nltk.tokenize import word_tokenize

    tokens = nltk.tokenize.word_tokenize(positive_word)
    from nltk.tokenize import word_tokenize
    from nltk.probability import FreqDist

    kemunculan = nltk.FreqDist(tokens)
    
    kemunculan2 = dict((k, v) for k, v in kemunculan.items() if v >= 4)
    
    # creating the dataset
    courses = list(kemunculan2.keys())
    values = list(kemunculan2.values())

    from collections import Counter
    positive_data['temp_list'] = positive_data['clean_tweets'].apply(lambda x:str(x).split())
    top = Counter([item for sublist in positive_data['temp_list'] for item in sublist])
    temp = pd.DataFrame(top.most_common(20))
    temp.columns = ['Common_words','count']
    temp.style.background_gradient(cmap='Blues')
    import plotly.express as px
    fig = px.bar(temp, x="count", y="Common_words", title='Commmon Words in positive sentiment', orientation='h', 
                width=700, height=700,color='Common_words')
    st.plotly_chart(fig)
#------END OF Barplot------#

col1, col2 = st.columns(2)
#------Sentiment Distribution------#
col2.title("Sentiment Distribution :bar_chart:")
col2.markdown("#####")
col2.text("")
col2.text("")
col2.text("")
col2.text("")
col2.text("")
col2.text("")


df_clean = df_clean.sort_values(by='date')
df_clean['date'] = pd.to_datetime(df_clean['date']).dt.date

df_clean['year']         = pd.DatetimeIndex(df_clean['date']).year
df_clean['month']        = pd.DatetimeIndex(df_clean['date']).month
df_clean['day']          = pd.DatetimeIndex(df_clean['date']).day
df_clean['day_of_year']  = pd.DatetimeIndex(df_clean['date']).dayofyear
df_clean['quarter']      = pd.DatetimeIndex(df_clean['date']).quarter
df_clean['season']       = df_clean.month%12 // 3 + 1
df_clean['date_MY'] = pd.to_datetime(df_clean[['year', 'month', 'day']])
df_label = df_clean[['date_MY', 'label']]
ctdf = (df_label.reset_index()
          .groupby(['label'], as_index=False)
          .count()
          # rename isn't strictly necessary here, it's just for readability
          .rename(columns={'index':'Jumlah Sentiment'})
       )
fig = px.bar(ctdf, y='Jumlah Sentiment', x='label', text_auto='10', color='label')

fig.update_layout(height=500)
fig.update_traces(textfont_size=14, textangle=45, textposition="outside", cliponaxis=False)
fig.update_yaxes(tickfont=dict(size=20))
fig.update_xaxes(tickfont=dict(size=20))
col2.plotly_chart(fig, use_container_width=True, height=900, width = 900)
#------END OF Sentiment Distribution------#


#------WordCloud------#

col1.title("Sentiment Wordcloud :thought_balloon:")
col1.markdown("#####")
wc_selection = ['Negative Sentiment', 'Positive Sentiment']

wc_result = col1.selectbox("Select your data", wc_selection, key = '1')

from wordcloud import WordCloud, ImageColorGenerator
from PIL import Image
st.set_option('deprecation.showPyplotGlobalUse', False)
mask = np.array(Image.open('logo-twitter-png-5860.jpg'))
# mask = np.array(Image.open('twitter_mask (1).png'))
if wc_result == 'Negative Sentiment':
    def generate_better_wordcloud(data, mask=None):
        cloud = WordCloud(scale=3,
                        max_words=10000,
                        colormap='RdYlGn',
                        mask=mask,
                        background_color='white',
                        collocations=True).generate_from_text(data)
        plt.figure(figsize=(6.5,6.5))
        plt.imshow(cloud)
        plt.axis('off')

    wc = generate_better_wordcloud(negative_word, mask=mask)
    file_name = 'negword_cloud'
    title = 'Negative Wordcloud'
    plt.savefig(f'{file_name}.png',bbox_inches='tight',pad_inches = 0)
    image = Image.open(f'{file_name}.png')
    col1.image(image, caption=f'{title}')
elif wc_result == 'Positive Sentiment':
    def generate_better_wordcloud(data, mask=None):
        cloud = WordCloud(scale=3,
                        max_words=10000,
                        colormap='RdYlGn',
                        mask=mask,
                        background_color='white',
                        collocations=True).generate_from_text(data)
        plt.figure(figsize=(6.5,6.5))
        plt.imshow(cloud)
        plt.axis('off')

    wc = generate_better_wordcloud(positive_word, mask=mask)
    file_name = 'posword_cloud'
    title = 'Positive Wordcloud'
    plt.savefig(f'{file_name}.png',bbox_inches='tight',pad_inches = 0)
    image = Image.open(f'{file_name}.png')
    col1.image(image, caption=f'{title}')

#------END OF WordCloud------#


# #------TIME SERIES PLOT------#
st.title("Time Series Plot 	:chart_with_upwards_trend:")
#Sorting And Feature Engineering
df_clean = df_clean.sort_values(by='date')
df_clean['date'] = pd.to_datetime(df_clean['date']).dt.date

df_clean['year']         = pd.DatetimeIndex(df_clean['date']).year
df_clean['month']        = pd.DatetimeIndex(df_clean['date']).month
df_clean['day']          = pd.DatetimeIndex(df_clean['date']).day
df_clean['day_of_year']  = pd.DatetimeIndex(df_clean['date']).dayofyear
df_clean['quarter']      = pd.DatetimeIndex(df_clean['date']).quarter
df_clean['season']       = df_clean.month%12 // 3 + 1



df_clean['date_MY'] = pd.to_datetime(df_clean[['year', 'month', 'day']]).dt.date

from datetime import datetime
# generate_time = st.button("Generate", key = '767')

# last_update_ts = df_last_date['last_update'].values[0]
# new_update_ts = df_new_date['last_update'].values[0]
# st.write(last_update_ts)
# start_date = datetime.strptime(last_update_ts, '%Y-%d-%m').date()
# end_date = datetime.strptime(new_update_ts, '%Y-%d-%m').date()
# st.write(start_date)

try:
    start_date, end_date = st.date_input('Input Date :', [])
    if start_date < end_date:
        pass
    else:
        st.error('Error: Date de fin doit être choisi après la dete de début.')
    mask = (df_clean['date_MY'] > start_date) & (df_clean['date_MY'] <= end_date)
    df_clean = df_clean.loc[mask]
    df_label = df_clean[['date_MY', 'label']]
    ctdf = (df_label.reset_index()
            .groupby(['date_MY','label'], as_index=False)
            .count()
            .rename(columns={'index':'ct'})
        )
    negative_data = ctdf.loc[ctdf['label'] == "negative"] 
    positive_data = ctdf.loc[ctdf['label'] == "positive"] 
    neutral_data = ctdf.loc[ctdf['label'] == "neutral"] 
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces
    fig.add_trace(
        go.Scatter(x=positive_data.date_MY, y=positive_data.ct, name="Positive"),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(x=negative_data.date_MY, y=negative_data.ct, name="Negative"),
        secondary_y=False,
    )
    # Add figure title
    fig.update_layout(
        title_text="Plot Time Series"
    )

    # Set x-axis title
    fig.update_xaxes(title_text="Tanggal Tweet")
    # Set y-axes titles
    fig.update_yaxes(title_text="<b>Jumlah</b> Sentiment", secondary_y=False)
    fig.update_yaxes(title_text="<b>Tanggal</b> Tanggan Tweet", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)
except:
    pass
#------END OF Time Series Plot------#



#------Text Clustering------#

#---MSQL CONNECTION---#
import mysql.connector
from sqlalchemy import create_engine
conn_clean = create_engine('mysql+mysqlconnector://root:sukses25@localhost:3306/clean_tweets')
conn_date = create_engine('mysql+mysqlconnector://root:sukses25@localhost:3306/update_date')
conn_raw = create_engine('mysql+mysqlconnector://root:sukses25@localhost:3306/raw_tweets')

clean_query = "SELECT * FROM clean_tweets"
df_clean = pd.read_sql(clean_query, conn_clean)
df_clean.to_csv('clean_tweets.csv')
df_clean['clean_tweets'] = df_clean['clean_tweets'].replace('', np.nan)
df_clean = df_clean.dropna(subset=['clean_tweets'])
df_clean = df_clean.drop_duplicates(subset=['tweet_id'])
df_clean.to_csv('clean_tweets.csv')

from nltk.tokenize import word_tokenize

def word_tokenize_wrapper(text):
  return word_tokenize(text)

df_clean['lda_token'] = df_clean['clean_tweets'].apply(word_tokenize_wrapper)


df_clean['lda_token'] = df_clean['clean_tweets'].apply(word_tokenize_wrapper)
negative_data = df_clean.loc[df_clean['label'] == "negative"]
positive_data = df_clean.loc[df_clean['label'] == "positive"]
neutral_data = df_clean.loc[df_clean['label'] == "neutral"]
#---END OF CONNECTION---#   



st.title("Sentiment Cluster :triangular_ruler:")
st.markdown("#####")
cl_selection = ['Negative Sentiment', 'Positive Sentiment']

cl_result = st.selectbox("Select your data", bp_selection, key = '3')

if cl_result == 'Negative Sentiment':
    tx = negative_data.copy()
    from sklearn import preprocessing
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans


    # initialize the vectorizer
    vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=5, max_df=0.95)
    # fit_transform applies TF-IDF to clean texts - we save the array of vectors in X
    X = vectorizer.fit_transform(tx['clean_tweets'])
    from sklearn.cluster import KMeans

    # initialize kmeans with 3 centroids
    kmeans = KMeans(n_clusters=3, random_state=42)
    # fit the model
    kmeans.fit(X)
    # store cluster labels in a variable
    clusters = kmeans.labels_
    from sklearn.decomposition import PCA

    # initialize PCA with 2 components
    pca = PCA(n_components=2, random_state=42)
    # pass our X to the pca and store the reduced vectors into pca_vecs
    pca_vecs = pca.fit_transform(X.toarray())
    # save our two dimensions into x0 and x1
    x0 = pca_vecs[:, 0]
    x1 = pca_vecs[:, 1]
    # assign clusters and pca vectors to our dataframe 
    tx['cluster'] = clusters
    tx['x0'] = x0
    tx['x1'] = x1
    cl_name = []
    def get_top_keywords(n_terms):
        """This function returns the keywords for each centroid of the KMeans"""
        df = pd.DataFrame(X.todense()).groupby(clusters).mean() # groups the TF-IDF vector by cluster
        terms = vectorizer.get_feature_names_out() # access tf-idf terms

        for i,r in df.iterrows():
            # st.write('\nCluster {}'.format(i))
            cl_name.append([','.join([terms[t] for t in np.argsort(r)[-n_terms:]])]) # for each row of the dataframe, find the n terms that have the highest tf idf score
                
    get_top_keywords(10)
    # map clusters to appropriate labels 
    cluster_map = {0: f"{cl_name[0]}", 1: f"{cl_name[1]}", 2: f"{cl_name[2]}"}
    # apply mapping
    tx['cluster'] = tx['cluster'].map(cluster_map)
    # set image size
    plt.figure(figsize=(12, 7))
    # set a title
    plt.title("Negative Tweet Clustering", fontdict={"fontsize": 18})
    # set axes names
    plt.xlabel("X0", fontdict={"fontsize": 16})
    plt.ylabel("X1", fontdict={"fontsize": 16})
    # create scatter plot with seaborn, where hue is the class used to group the data
    sns.scatterplot(data=tx, x='x0', y='x1', hue='cluster', palette="viridis")
    file_name = 'neg_clus'
    # title = ''
    plt.savefig(f'{file_name}.png',bbox_inches='tight',pad_inches = 0)
    image = Image.open(f'{file_name}.png')

    left, middle, right = st.columns((2, 5, 2))
    with middle:
        st.image(image)#, caption=f'{title}')
elif cl_result == 'Positive Sentiment':
    tx = positive_data.copy()
    from sklearn import preprocessing
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans


    # initialize the vectorizer
    vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=5, max_df=0.95)
    # fit_transform applies TF-IDF to clean texts - we save the array of vectors in X
    X = vectorizer.fit_transform(tx['clean_tweets'])
    from sklearn.cluster import KMeans

    # initialize kmeans with 3 centroids
    kmeans = KMeans(n_clusters=3, random_state=42)
    # fit the model
    kmeans.fit(X)
    # store cluster labels in a variable
    clusters = kmeans.labels_
    from sklearn.decomposition import PCA

    # initialize PCA with 2 components
    pca = PCA(n_components=2, random_state=42)
    # pass our X to the pca and store the reduced vectors into pca_vecs
    pca_vecs = pca.fit_transform(X.toarray())
    # save our two dimensions into x0 and x1
    x0 = pca_vecs[:, 0]
    x1 = pca_vecs[:, 1]
    # assign clusters and pca vectors to our dataframe 
    tx['cluster'] = clusters
    tx['x0'] = x0
    tx['x1'] = x1
    cl_name = []
    def get_top_keywords(n_terms):
        """This function returns the keywords for each centroid of the KMeans"""
        df = pd.DataFrame(X.todense()).groupby(clusters).mean() # groups the TF-IDF vector by cluster
        terms = vectorizer.get_feature_names_out() # access tf-idf terms

        for i,r in df.iterrows():
            # st.write('\nCluster {}'.format(i))
            cl_name.append([','.join([terms[t] for t in np.argsort(r)[-n_terms:]])]) # for each row of the dataframe, find the n terms that have the highest tf idf score
                
    get_top_keywords(10)
    # map clusters to appropriate labels 
    cluster_map = {0: f"{cl_name[0]}", 1: f"{cl_name[1]}", 2: f"{cl_name[2]}"}
    # apply mapping
    tx['cluster'] = tx['cluster'].map(cluster_map)
    # set image size
    plt.figure(figsize=(12, 7))
    # set a title
    plt.title("Positive Tweet Clustering", fontdict={"fontsize": 18})
    # set axes names
    plt.xlabel("X0", fontdict={"fontsize": 16})
    plt.ylabel("X1", fontdict={"fontsize": 16})
    # create scatter plot with seaborn, where hue is the class used to group the data
    sns.scatterplot(data=tx, x='x0', y='x1', hue='cluster', palette="viridis")
    file_name = 'pos_clus'
    # title = ''
    plt.savefig(f'{file_name}.png',bbox_inches='tight',pad_inches = 0)
    image = Image.open(f'{file_name}.png')
    left, middle, right = st.columns((2, 5, 2))
    with middle:
        st.image(image)#, caption=f'{title}')

#------END OF CLUSTERING------#
