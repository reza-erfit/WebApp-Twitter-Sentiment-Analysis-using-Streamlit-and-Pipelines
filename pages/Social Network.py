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
df_clean = pd.read_sql(clean_query, conn_clean).head(10000)
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

page_title = 'Social Network Analysis'
page_icon = ":pushpin:" 
st.set_page_config(page_title=page_title, page_icon=page_icon, layout='wide')


#------Social Network------#

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import re
import numpy as np
from collections import Counter
plt.style.use('ggplot')
from bokeh.io import output_notebook, show, save
from bokeh.models import Range1d, Circle, ColumnDataSource, MultiLine
from bokeh.plotting import figure
from bokeh.plotting import from_networkx
from bokeh.models import Range1d, Circle, ColumnDataSource, MultiLine, EdgesAndLinkedNodes, NodesAndLinkedEdges
from bokeh.palettes import Blues8, Reds8, Purples8, Oranges8, Viridis8, Spectral8
from bokeh.models import Range1d, Circle, ColumnDataSource, MultiLine, EdgesAndLinkedNodes, NodesAndLinkedEdges, LabelSet
import networkx

st.title("Social Network Analysis")

tm_selection = ['Negative Sentiment', 'Positive Sentiment']
bp_selection = ['Negative Sentiment', 'Positive Sentiment']
tm_result = st.selectbox("Select your data", bp_selection, key = '699')

if tm_result == 'Negative Sentiment':
    st.subheader("Data Negative")
    negative_data=negative_data[['user', 'mentioned_user','followers_count', 'statuses_count',"location"]]
    negative_data = negative_data.dropna()
    ctdf = (negative_data.reset_index()
            .groupby(['user','mentioned_user','followers_count', 'statuses_count','location'], as_index=False)
            .count()
            # rename isn't strictly necessary here, it's just for readability
            .rename(columns={'index':'weight'})
        )
    edges = ctdf.groupby(['user','mentioned_user','followers_count','statuses_count','location']).agg({'weight':'sum'}).reset_index().query('weight > 0')
    edges = edges.sort_values('weight',ascending=False)
    st.dataframe(edges)
    st.subheader("Plot Twitter Network Analysis Negative")
    # st.dataframe(edges)

    G = networkx.from_pandas_edgelist(edges, 'user', 'mentioned_user', 'weight')
    #Choose a title!
    title = 'Network Analysis of Twitter'

    #Establish which categories will appear when hovering over each node
    HOVER_TOOLTIPS = [("Character", "@index")]

    #Create a plot — set dimensions, toolbar, and title
    plot = figure(tooltips = HOVER_TOOLTIPS,
                tools="pan,wheel_zoom,save,reset", active_scroll='wheel_zoom',
            x_range=Range1d(-10.1, 10.1), y_range=Range1d(-10.1, 10.1), title=title)

    network_graph = from_networkx(G, networkx.spring_layout, scale=10, center=(0, 0))

    #Set node size and color
    network_graph.node_renderer.glyph = Circle(size=15, fill_color='skyblue')

    #Set edge opacity and width
    network_graph.edge_renderer.glyph = MultiLine(line_alpha=0.5, line_width=1)
    x, y = zip(*network_graph.layout_provider.graph_layout.values())
    node_labels = list(G.nodes())
    source = ColumnDataSource({'x': x, 'y': y, 'name': [node_labels[i] for i in range(len(x))]})
    labels = LabelSet(x='x', y='y', text='name', source=source, background_fill_color='white', text_font_size='10px', background_fill_alpha=.7)
    plot.renderers.append(labels)

    #Add network graph to the plot
    plot.renderers.append(network_graph)

    # show(plot)
    #save(plot, filename=f"{title}.html")
    # st.title("Plot Social Network Analysis :thought_balloon:")
    st.bokeh_chart(plot, use_container_width=True)

elif tm_result == 'Positive Sentiment':
    st.subheader("Data Positive")
    positive_data = positive_data[['user', 'mentioned_user','followers_count', 'statuses_count',"location"]]
    positive_data = positive_data.dropna()
    ctdf = (positive_data.reset_index()
            .groupby(['user','mentioned_user','followers_count', 'statuses_count','location'], as_index=False)
            .count()
            # rename isn't strictly necessary here, it's just for readability
            .rename(columns={'index':'weight'})
        )
    edges = ctdf.groupby(['user','mentioned_user','followers_count','statuses_count','location']).agg({'weight':'sum'}).reset_index().query('weight > 0')
    edges = edges.sort_values('weight',ascending=False)
    st.dataframe(edges)

    # st.dataframe(edges)
    st.subheader("Plot Twitter Network Analysis Positive")
    G = networkx.from_pandas_edgelist(edges, 'user', 'mentioned_user', 'weight')
    #Choose a title!
    title = 'Network Analysis of Twitter'

    #Establish which categories will appear when hovering over each node
    HOVER_TOOLTIPS = [("Character", "@index")]

    # Create a plot — set dimensions, toolbar, and title
    plot = figure(tooltips = HOVER_TOOLTIPS,
                tools="pan,wheel_zoom,save,reset", active_scroll='wheel_zoom',
            x_range=Range1d(-10.1, 10.1), y_range=Range1d(-10.1, 10.1), title=title)

    network_graph = from_networkx(G, networkx.spring_layout, scale=10, center=(0, 0))

    #Set node size and color
    network_graph.node_renderer.glyph = Circle(size=15, fill_color='skyblue')

    #Set edge opacity and width
    network_graph.edge_renderer.glyph = MultiLine(line_alpha=0.5, line_width=1)
    x, y = zip(*network_graph.layout_provider.graph_layout.values())
    node_labels = list(G.nodes())
    source = ColumnDataSource({'x': x, 'y': y, 'name': [node_labels[i] for i in range(len(x))]})
    labels = LabelSet(x='x', y='y', text='name', source=source, background_fill_color='white', text_font_size='10px', background_fill_alpha=.7)
    plot.renderers.append(labels)

    #Add network graph to the plot
    plot.renderers.append(network_graph)

    # show(plot)
    #save(plot, filename=f"{title}.html")
    # st.title("Plot Social Network Analysis :thought_balloon:")
    st.bokeh_chart(plot, use_container_width=True)
    # st.dataframe(positive_data)


#------End Of Social Network------#
