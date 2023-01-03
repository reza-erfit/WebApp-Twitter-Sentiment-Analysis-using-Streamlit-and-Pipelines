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
df_clean = pd.read_sql(clean_query, conn_clean)
df_clean['clean_tweets'] = df_clean['clean_tweets'].replace('', np.nan)
df_clean = df_clean.dropna(subset=['clean_tweets'])
df_clean = df_clean.drop_duplicates(subset=['tweet_id'])


from nltk.tokenize import word_tokenize

def word_tokenize_wrapper(text):
  return word_tokenize(text)

df_clean['lda_token'] = df_clean['clean_tweets'].apply(word_tokenize_wrapper)


date_query = "SELECT * FROM update_date ORDER BY last_update DESC LIMIT 1"
df_last_date = pd.read_sql(date_query, conn_date)

#---END OF CONNECTION---#  
page_title = 'Sentiment Scrapper'
page_icon = ":scissors:" 
st.set_page_config(page_title=page_title, page_icon=page_icon, layout='wide')

# st.sidebar.markdown("# Sidebar ")

#------SCRAPPER------#

st.sidebar.header("KELOMPOK 9")
st.title(page_title + " " + page_icon)
update = st.button("Update Data")
number_of_tweets = st.number_input('Number of tweets', min_value=1, max_value=10000000, value=1, step=1)#st.slider("Number of tweet",min_value=1, max_value=100000)

# st.write("Loading")
t = st.empty()

if update:
    from datetime import datetime
    current_date = datetime.today().strftime('%Y-%m-%d')

    last_update = df_last_date['last_update'].values[0]
    # query = f"resesi until:{current_date} since:{last_update}"
    query = f"resesi until:{current_date} since:2020-01-01"
    query = 'resesi'
    tweets = []
    limits = number_of_tweets
    i = 1
    for tweet in sntwitter.TwitterSearchScraper(query).get_items():
        if len(tweets) == limits:
            break
        if tweet.mentionedUsers == None:
            tweets.append([tweet.url,tweet.date, tweet.user.username, tweet.content, tweet.id, tweet.user.followersCount, tweet.user.friendsCount, tweet.user.statusesCount, tweet.user.location, tweet.replyCount, tweet.retweetCount, tweet.likeCount, tweet.mentionedUsers])
            t.markdown(i)
            i+=1
        else:
            tweets.append([tweet.url,tweet.date, tweet.user.username, tweet.content, tweet.id, tweet.user.followersCount, tweet.user.friendsCount, tweet.user.statusesCount, tweet.user.location, tweet.replyCount, tweet.retweetCount, tweet.likeCount, tweet.mentionedUsers[0].username])
            # print(vars(tweet))
            t.markdown(i)
            i+=1

    dfs = pd.DataFrame(tweets, columns = ['url','date', 'user', 'tweet', "tweet_id", "followers_count", "friends_count", "statuses_count", "location", "reply_count", "retweet_count", "like_count", "mentioned_user"])
    dfs['tweet'] = dfs['tweet'].astype(str)
    # Saving raw data to sql database
    dfs.to_sql(con = conn_raw, name = 'raw_tweets', if_exists = 'append', index = False)

    # Saving update date to sql database    
    df_date = pd.DataFrame([current_date], columns=['last_update'])
    df_date.to_sql(con = conn_date, name = 'update_date', if_exists = 'append', index = False)
    
    st.write(dfs)




    st.title("IndoBERT Sentiment Labelling 	:pencil:")
  
    #------IndoBERT------#
    from transformers import pipeline
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    pretrained= "mdhugol/indonesia-bert-sentiment-classification"

    model = AutoModelForSequenceClassification.from_pretrained(pretrained)
    tokenizer = AutoTokenizer.from_pretrained(pretrained)

    sentiment_analysis = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

    label_index = {'LABEL_0': 'positive', 'LABEL_1': 'neutral', 'LABEL_2': 'negative'}


    labell = []
    scoree = []
    t = st.empty()
    ct = 1
    for i in dfs['tweet']:
        
        result = sentiment_analysis(i)
        status = label_index[result[0]['label']]
        score = result[0]['score']
        labell.append(status)
        scoree.append(score)
        t.markdown(ct)
        ct+=1

    label = pd.DataFrame(labell, columns = ['label'])
    score = pd.DataFrame(scoree, columns = ['score'])
    ab = pd.concat([score, label], axis=1)
    # st.write(ab)
    df2 = pd.concat([dfs,ab],axis = 1)
    st.write(df2)
    # st.write(df2)
    #------END OF IndoBERT------#

    # df2 = df_clean.copy()
    #------CLEANING------#
    import re
    import string
    def processclean(text): 
        text = str(text).lower() # text menjadi lowecase
        text = text.encode('ascii', 'ignore').decode()# Hapus karakter unicode
        text = re.sub(r"\w+\.co\.id","",text) # menghapus link .co.id
        text = re.sub(r"\w+\.com","",text) # menghapus link .com
        text = re.sub(r"\(.*\)","",text) # menghapus teks yang ada pada dalam kurung
        text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text) # Hapus tanda baca
        text = re.sub(r'\b\w{1}\b', '', text) # menghapus teks dengan karakter <=1
        text = re.sub("\S*\d\S*", "", text).strip() #remove digit from string
        text = re.sub(r"\b\d+\b", " ", text) #remove digit or numbers
        text = re.sub(r'\n', ' ', text) # menghapus new line
        text = re.sub(r'\s+',' ',text) # menghapus extra space
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r"(?:\@|https?\://)\S+", "", text)
        return text

    df2['clean'] = df2['tweet'].apply(processclean)

    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize

    def word_tokenize_wrapper(text):
        return word_tokenize(text)

    df2['clean_token'] = df2['clean'].apply(word_tokenize_wrapper)

    newWord=['wkwk','skrg','blm','lg','di','sy','dtg','khan','si','kl','yg','jd','klo','pa','nya','tp','ga','jg','https','co','aja','ya','gw','kalo','tuh','tau','gk','gak','kalo','amp','gitu','krn','dr','sih','gue','bgt','aja','ya','krn','pake','udah','sampe','udah','emang','nggak','gk','udh','kela','duanya','banget','tdk','semoga','dgn','nih','loh','dpt','yaa','dah','kak','sm','ngga','dg','deh','lho','utk','kali','sahabat', 'nomor','yg','jd','klo','pa','nya','tp','ga','jg','https','co','aja','ya','gw','org','kalo','tuh','tau','gk','gak','kalo','amp','gitu','krn','dr','sih','gue','bgt','aja','ya','krn','pake','udah','sampe','udah','emang','nggak','gk','udh','kela','duanya','banget','tdk','semoga','dgn','nih','loh','dpt','yaa','dah','kak','sm','ngga','dg','deh','lho','utk','kali', 'a', 'b','c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v','w','x','y','z', 'gua','sya', 'iya', 'ni','lu',
            'loe', 'mah','resesi','gmn','aga','gini','lgi']
    list_stopwords = stopwords.words('indonesian')
    list_stopwords = list_stopwords+[i for i in newWord]
    list_stopwords = set(list_stopwords)

    def remove_stopwords(words):
        return [word for word in words if word not in list_stopwords]

    df2['clean_v1'] = df2['clean_token'].apply(remove_stopwords)
    df2['clean_tweets'] = df2['clean_v1'].agg(lambda x: ' '.join(map(str, x)))
    df2 = df2.drop('clean_token', axis = 1)
    df2 = df2.drop('clean_v1', axis = 1)
    df2 = df2.drop('clean', axis = 1)

    # Saving update date to sql database    
    df2.to_sql(con = conn_clean, name = 'clean_tweets', if_exists = 'append', index = False)
    #------END OF CLEANING------#