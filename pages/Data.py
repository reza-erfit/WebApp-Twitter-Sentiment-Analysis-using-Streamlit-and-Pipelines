
import streamlit as st
import streamlit as st
import plotly.graph_objects as go
import snscrape.modules.twitter as sntwitter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#---MSQL CONNECTION---#
import mysql.connector
from sqlalchemy import create_engine
conn_clean = create_engine('mysql+mysqlconnector://root:sukses25@localhost:3306/clean_tweets')
conn_date = create_engine('mysql+mysqlconnector://root:sukses25@localhost:3306/update_date')
conn_raw = create_engine('mysql+mysqlconnector://root:sukses25@localhost:3306/raw_tweets')

clean_query = "SELECT * FROM clean_tweets"
df_clean = pd.read_sql(clean_query, conn_clean) #.head(30000)
df_clean['clean_tweets'] = df_clean['clean_tweets'].replace('', np.nan)
df_clean = df_clean.dropna(subset=['clean_tweets'])
df_clean = df_clean.drop_duplicates(subset=['tweet_id'])
df_clean.to_csv("clean_tweets.csv")
page_title = 'Recession Sentiment Dashboard'
page_icon = ":bar_chart:"
st.set_page_config(page_title=page_title, page_icon=page_icon, layout='wide')

# st.sidebar.markdown("# Sidebar")
st.sidebar.header("KELOMPOK 9")
st.title("Data 	:books:")

wc_selection = ['All Data','Negative Sentiment', 'Positive Sentiment']

wc_result = st.selectbox("Select your data", wc_selection, key = '1')

if wc_result == 'All Data':
    st.write(df_clean)
elif wc_result == 'Negative Sentiment':
    df_clean_negative = df_clean.loc[df_clean['label'] == 'negative']
    st.write(df_clean_negative)
elif wc_result == 'Positive Sentiment':
    df_clean_positive = df_clean.loc[df_clean['label'] == 'positive']
    st.write(df_clean_positive)


