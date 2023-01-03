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


from nltk.tokenize import word_tokenize

def word_tokenize_wrapper(text):
  return word_tokenize(text)

df_clean['lda_token'] = df_clean['clean_tweets'].apply(word_tokenize_wrapper)


date_query = "SELECT * FROM update_date ORDER BY last_update DESC LIMIT 1"
df_last_date = pd.read_sql(date_query, conn_date)

negative_data = df_clean.loc[df_clean['label'] == "negative"]
positive_data = df_clean.loc[df_clean['label'] == "positive"]
neutral_data = df_clean.loc[df_clean['label'] == "neutral"]

#---END OF CONNECTION---#  
page_title = 'Recession Sentiment Scrapper'
page_icon = ":bar_chart:" 
st.set_page_config(page_title=page_title, page_icon=page_icon, layout='wide')

# st.sidebar.markdown("# Sidebar ")


#------LDA Topic Modelling------#
st.title("Sentiment Topic Modelling :pushpin:")
st.markdown("#####")
tm_selection = ['Negative Sentiment', 'Positive Sentiment']
bp_selection = ['Negative Sentiment', 'Positive Sentiment']
st.subheader("Coherence vs Num of Topic")
tm_result = st.selectbox("Select your data", bp_selection, key = '69')

from PIL import Image
import gensim
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary
from numpy import array
from gensim import corpora, models

if tm_result == 'Negative Sentiment':

    dictionary = gensim.corpora.Dictionary(negative_data['lda_token'])
    count = 0
    for k, v in dictionary.iteritems():
        # print(k, v)
        count += 1
        if count > 10:
            break
    bow_corpus = [dictionary.doc2bow(doc) for doc in negative_data['lda_token']]
    # bow_corpus[999]
    # bow_corpus
    bow_doc_999 = bow_corpus[1]
    # st.write(bow_corpus)
    # for i in range(len(bow_doc_999)):
    #     print("Word {} (\"{}\") appears {} time.".format(bow_doc_999[i][0], 
    #                                             dictionary[bow_doc_999[i][0]], 
    # bow_doc_999[i][1]))


    tfidf = models.TfidfModel(bow_corpus)
    corpus_tfidf = tfidf[bow_corpus]
    from pprint import pprint
    for doc in corpus_tfidf:
        # st.write(doc)

        break

    #function to compute coherence values

    # def compute_coherence_values(dictionary, corpus, texts, limit, start, step):
    #     coherence_values = []
    #     model_list = []

    #     for num_topics in range(start, limit, step):
    #         model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, iterations=100)
    #         model_list.append(model)
    #         coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
    #         coherence_values.append(coherencemodel.get_coherence())

    #     return model_list, coherence_values
    # import sys
    # sys.path.append(r'C:/Users/LENOVO/NLP/packages')
    # from lda_lib import compute_coherence_values
    # start=1
    # limit=5
    # step=1
    # model_list, coherence_values = compute_coherence_values(dictionary, corpus=bow_corpus, 
    #                                                         texts=positive_data['lda_token'], start=start, 
    #                                                          limit=limit, step=step)
    start=1
    limit=5
    step=1
    coherence_values = [0.30503694083207583,
 0.32915379631891106,
 0.3127301876426134,
 0.32228732785545733]
    #show graphs
    x = range(start, limit, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics for Negative Sentiment")
    plt.ylabel("Coherence Score")
    plt.legend(("coherence_values"), loc='best')
    file_name = 'neg_top'
    # title = ''
    plt.savefig(f'{file_name}.png',bbox_inches='tight',pad_inches = 0)
    image = Image.open(f'{file_name}.png')
    
    st.image(image)#, caption=f'{title}')






    st.text("")
    st.text("")
    st.subheader("Generate Topic")

    generate = st.button("Generate Topic", key = '1')
    number_of_topic = st.number_input('Number of topic', min_value=1, max_value=100, value=1, step=1)
    st.text("")
    st.text("")
    if generate:
        st.subheader("Coherence Score")
        for m, cv in zip(x, coherence_values):
            model = LdaModel(corpus=bow_corpus, id2word=dictionary, num_topics=number_of_topic)#num topic menyesuaikan hasil dari coherence value paling tinggi
            st.write("Num Topics =", m, " has Coherence Value of", round(cv, 6))
        
        st.text("")
        st.text("")
        st.subheader("Topic Candidate")
        for idx, topic in model.print_topics(-1):
            top_words_per_topic = []
            st.write('Topic: {} Word: {}'.format(idx, topic))
        st.text("")
        st.text("")
        st.subheader("Word in Topic")
        for t in range(model.num_topics):
            top_words_per_topic.extend([(t, ) + x for x in model.show_topic(t, topn = number_of_topic)])
        #pd.DataFrame(top_words_per_topic, columns=['Topic', 'Word', 'P']).to_csv("top_words.csv")
        top_word = pd.DataFrame(top_words_per_topic, columns=['Topic', 'Word','P'])
        st.write(top_word)

        import gensim
        import pyLDAvis.gensim_models
        import pyLDAvis
        data = pyLDAvis.gensim_models.prepare(model, bow_corpus, dictionary)
        pyLDAvis.save_html(data, 'resesi-Negative-lda-gensim-bow.html')
        import streamlit.components.v1 as components

        st.subheader("LDA Visualization")

        HtmlFile = open("resesi-Negative-lda-gensim-bow.html", 'r', encoding='utf-8')
        source_code = HtmlFile.read() 
        components.html(source_code,height = 800, width = 1920)


elif tm_result == 'Positive Sentiment':
    dictionary = gensim.corpora.Dictionary(positive_data['lda_token'])
    count = 0
    for k, v in dictionary.iteritems():
        # print(k, v)
        count += 1
        if count > 10:
            break
    bow_corpus = [dictionary.doc2bow(doc) for doc in positive_data['lda_token']]
    # bow_corpus[999]
    # bow_corpus
    bow_doc_999 = bow_corpus[1]
    # st.write(bow_corpus)
    # for i in range(len(bow_doc_999)):
    #     print("Word {} (\"{}\") appears {} time.".format(bow_doc_999[i][0], 
    #                                             dictionary[bow_doc_999[i][0]], 
    # bow_doc_999[i][1]))


    tfidf = models.TfidfModel(bow_corpus)
    corpus_tfidf = tfidf[bow_corpus]
    from pprint import pprint
    for doc in corpus_tfidf:
        # st.write(doc)

        break

    #function to compute coherence values

    # def compute_coherence_values(dictionary, corpus, texts, limit, start, step):
    #     coherence_values = []
    #     model_list = []

    #     for num_topics in range(start, limit, step):
    #         model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, iterations=100)
    #         model_list.append(model)
    #         coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
    #         coherence_values.append(coherencemodel.get_coherence())

    #     return model_list, coherence_values
    # import sys
    # sys.path.append(r'C:/Users/LENOVO/NLP/packages')
    # from lda_lib import compute_coherence_values
    start=1
    limit=5
    step=1
    # model_list, coherence_values = compute_coherence_values(dictionary, corpus=bow_corpus, 
    #                                                         texts=positive_data['lda_token'], start=start, 
    #                                                         limit=limit, step=step)
    coherence_values = [0.3188582674026258,
 0.33402102508972,
 0.29533151252812195,
 0.29853166171760226]
    #show graphs
    x = range(start, limit, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics for Positive Sentiment")
    plt.ylabel("Coherence Score")
    plt.legend(("coherence_values"), loc='best')
    file_name = 'pos_top'
    # title = ''
    plt.savefig(f'{file_name}.png',bbox_inches='tight',pad_inches = 0)
    image = Image.open(f'{file_name}.png')
    st.image(image)#, caption=f'{title}')






    st.text("")
    st.text("")
    
    st.subheader("Generate Topic")
    generate = st.button("Generate Topic", key = '2')
    number_of_topic = st.number_input('Number of topic', min_value=1, max_value=100, value=1, step=1)
    st.text("")
    st.text("")
    if generate:
        st.subheader("Coherence Score")
        for m, cv in zip(x, coherence_values):
            model = LdaModel(corpus=bow_corpus, id2word=dictionary, num_topics=number_of_topic)#num topic menyesuaikan hasil dari coherence value paling tinggi
            st.write("Num Topics =", m, " has Coherence Value of", round(cv, 6))
        
        st.text("")
        st.text("")
        st.subheader("Topic Candidate")
        for idx, topic in model.print_topics(-1):
            top_words_per_topic = []
            st.write('Topic: {} Word: {}'.format(idx, topic))
        st.text("")
        st.text("")
        st.subheader("Word in Topic")
        for t in range(model.num_topics):
            top_words_per_topic.extend([(t, ) + x for x in model.show_topic(t, topn = number_of_topic)])
        #pd.DataFrame(top_words_per_topic, columns=['Topic', 'Word', 'P']).to_csv("top_words.csv")
        top_word = pd.DataFrame(top_words_per_topic, columns=['Topic', 'Word','P'])
        st.write(top_word)
        import gensim
        import pyLDAvis.gensim_models
        import pyLDAvis
        data = pyLDAvis.gensim_models.prepare(model, bow_corpus, dictionary)
        print(data)
        pyLDAvis.save_html(data, 'resesi-Positive-lda-gensim-bow.html')
        import streamlit.components.v1 as components

        st.subheader("LDA Visualization")

        HtmlFile = open("resesi-Positive-lda-gensim-bow.html", 'r', encoding='utf-8')
        source_code = HtmlFile.read() 
        print(source_code)
        components.html(source_code,height = 800, width = 1920)