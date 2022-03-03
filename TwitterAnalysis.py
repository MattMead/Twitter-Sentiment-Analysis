#!/usr/bin/env python
# coding: utf-8

# In[87]:


import sklearn
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import seaborn as sns
from plotnine import * 

import warnings
warnings.filterwarnings('ignore')


# ## Loading and Cleaning Tweets Dataset

# In[88]:


tweets = pd.read_csv("~/Documents/CPSC_Courses/CPSC 392/datasets/Tweets.csv")
tweets.head()


# In[89]:


tweets.info()


# I am now looking at size of the dataset, along with what parameters I am working with and the types of each

# Lets now look at whether or not there is missing data within this dataset. To do this, I will count all of the na's in each column

# In[90]:


tweets.isna().sum()


# From this we can see that there is missing data in the following variables: "negativereason", "negativereason_confidence", "airline_sentiment_gold", "negativereason_gold", "tweet_coord", "tweet_location", and "user_timezone". I will remove  "airline_sentiment_gold", "negativereason_gold", "tweet_coord" because they have an absurd amount of missing values.

# In[91]:


tweets.drop(['airline_sentiment_gold','negativereason_gold', 'tweet_coord'], axis=1, inplace=True)


# In[92]:


tweets.info()


# As you can see, these columns have now been deleted. 

# ## Data Visualizations

# Lets first start by seeing how many tweets there are for each airline

# In[93]:


colors=sns.color_palette("pastel", 10) 
pd.Series(tweets["airline"]).value_counts().plot(kind = "bar",width= 0.7,color=colors, edgecolor='black',figsize=(10,8),fontsize=10,rot = 0)
plt.xlabel('Airlines', fontsize=12)
plt.ylabel('Number of Tweets', fontsize=12)
plt.title("Number of Tweets for each Airlines", fontsize = 16, fontweight = 'bold')


# As we can see, the airline with the most amount of tweets about them in this dataset is United. Lets see the proportions.

# In[94]:


colors=sns.color_palette("pastel", 10)
pd.Series(tweets["airline"]).value_counts().plot(kind="pie",colors=colors,labels=["United", "US Airways", "American","Southwest", "Delta", "Virgin America",]
,autopct='%.2f', fontsize=12,figsize=(8, 8),title = "Proportion of Tweets for Each Airline")


# In[95]:


colors=sns.color_palette("pastel", 10) 
pd.Series(tweets["airline_sentiment"]).value_counts().plot(kind = "bar",width= 0.7,color=colors, edgecolor='black',figsize=(10,8),fontsize=10,rot = 0)
plt.xlabel('Sentiments', fontsize=12)
plt.ylabel('Sentiment Frequency', fontsize=12)
plt.title("Frequency of Sentiments", fontsize = 16, fontweight = 'bold')


# In[96]:


colors=sns.color_palette("pastel", 10)
pd.Series(tweets["airline_sentiment"]).value_counts().plot(kind="pie",colors=colors,labels=["negative", "neutral", "positive",]
,autopct='%.2f', fontsize=12,figsize=(8, 8),title = "Proportion of Tweets for Each Sentiment")


# As shown above, there is a higher frequency of negative tweets about airlines then there are neutral and positive. This may indicate that when people tweet about airlines, it is usually to say something negative. Lets try doing this once again except lets see what it looks like for each particular airline.

# In[97]:


Airways = tweets.loc[tweets['airline'] == 'US Airways']
colors=sns.color_palette("pastel", 10) 
pd.Series(Airways["airline_sentiment"]).value_counts().plot(kind = "bar",width= 0.7,color=colors, edgecolor='black',figsize=(10,8),fontsize=10,rot = 0)
plt.xlabel('Sentiments', fontsize=12)
plt.ylabel('Frequency of Sentiments', fontsize=12)
plt.title("Frequency of Sentiments: US Airways", fontsize = 16, fontweight = 'bold')


# In[98]:


United = tweets.loc[tweets['airline'] == 'United']
colors=sns.color_palette("pastel", 10) 
pd.Series(United["airline_sentiment"]).value_counts().plot(kind = "bar",width= 0.7,color=colors, edgecolor='black',figsize=(10,8),fontsize=10,rot = 0)
plt.xlabel('Sentiments', fontsize=12)
plt.ylabel('Frequency of Sentiments', fontsize=12)
plt.title("Frequency of Sentiments: United", fontsize = 16, fontweight = 'bold')


# In[99]:


American = tweets.loc[tweets['airline'] == 'American']
colors=sns.color_palette("pastel", 10) 
pd.Series(American["airline_sentiment"]).value_counts().plot(kind = "bar",width= 0.7,color=colors, edgecolor='black',figsize=(10,8),fontsize=10,rot = 0)
plt.xlabel('Sentiments', fontsize=12)
plt.ylabel('Frequency of Sentiments', fontsize=12)
plt.title("Frequency of Sentiments: American", fontsize = 16, fontweight = 'bold')


# In[100]:


Southwest = tweets.loc[tweets['airline'] == 'Southwest']
colors=sns.color_palette("pastel", 10) 
pd.Series(Southwest["airline_sentiment"]).value_counts().plot(kind = "bar",width= 0.7,color=colors, edgecolor='black',figsize=(10,8),fontsize=10,rot = 0)
plt.xlabel('Sentiments', fontsize=12)
plt.ylabel('Frequency of Sentiments', fontsize=12)
plt.title("Frequency of Sentiments: Southwest", fontsize = 16, fontweight = 'bold')


# In[101]:


Delta = tweets.loc[tweets['airline'] == 'Delta']
colors=sns.color_palette("pastel", 10) 
pd.Series(Delta["airline_sentiment"]).value_counts().plot(kind = "bar",width= 0.7,color=colors, edgecolor='black',figsize=(10,8),fontsize=10,rot = 0)
plt.xlabel('Sentiments', fontsize=12)
plt.ylabel('Frequency of Sentiments', fontsize=12)
plt.title("Frequency of Sentiments: Delta", fontsize = 16, fontweight = 'bold')


# In[102]:


Virgin = tweets.loc[tweets['airline'] == 'Virgin America']
colors=sns.color_palette("pastel", 10) 
pd.Series(Virgin["airline_sentiment"]).value_counts().plot(kind = "bar",width= 0.7,color=colors, edgecolor='black',figsize=(10,8),fontsize=10,rot = 0)
plt.xlabel('Sentiments', fontsize=12)
plt.ylabel('Frequency of Sentiments', fontsize=12)
plt.title("Frequency of Sentiments: Virgin", fontsize = 16, fontweight = 'bold')


# As we can see, the tweet sentiment for Virgin America, Delta, and Southwest the highest. But the tweet sentiments for American, United, and US Airways are not to great. I am not very surprised that united is the leader for the most negative tweets due to all the news I have heard about their poor customer service. Lets now look at the proportion of the negative tweets for each airline.

# Method for making a pie chart for each airline.

# In[103]:


colors=sns.color_palette("pastel", 10)
pd.Series(Virgin["airline_sentiment"]).value_counts().plot(kind="pie",colors=colors,labels=["Negative", "Neutral", "Positive"],
autopct='%.2f', fontsize=12,figsize=(6, 6),title = "Proportion of Sentiment for Virgin America")


# In[104]:


colors=sns.color_palette("pastel", 10)
pd.Series(Delta["airline_sentiment"]).value_counts().plot(kind="pie",colors=colors,labels=["Negative", "Neutral", "Positive"],
autopct='%.2f', fontsize=12,figsize=(6, 6),title = "Proportion of Sentiment for Delta")


# In[105]:


colors=sns.color_palette("pastel", 10)
pd.Series(Southwest["airline_sentiment"]).value_counts().plot(kind="pie",colors=colors,labels=["Negative", "Neutral", "Positive"],
autopct='%.2f', fontsize=12,figsize=(6, 6),title = "Proportion of Sentiment for Southwest")


# In[106]:


colors=sns.color_palette("pastel", 10)
pd.Series(American["airline_sentiment"]).value_counts().plot(kind="pie",colors=colors,labels=["Negative", "Neutral", "Positive"],
autopct='%.2f', fontsize=12,figsize=(6, 6),title = "Proportion of Sentiment for American")


# In[107]:


colors=sns.color_palette("pastel", 10)
pd.Series(United["airline_sentiment"]).value_counts().plot(kind="pie",colors=colors,labels=["Negative", "Neutral", "Positive"],
autopct='%.2f', fontsize=12,figsize=(6, 6),title = "Proportion of Sentiment for United")


# In[108]:


colors=sns.color_palette("pastel", 10)
pd.Series(Airways["airline_sentiment"]).value_counts().plot(kind="pie",colors=colors,labels=["Negative", "Neutral", "Positive"],
autopct='%.2f', fontsize=12,figsize=(6, 6),title = "Proportion of Sentiment for US Airways")


# From these pie charts, we can see the the airline with the highest proportion of negative tweets is US Airways. The airline with the highest proportion of posotive tweets is Virgin America. We will now go into the specifics of the tweets, looking at what the negative reason are.

# In[109]:


tweets.negativereason.value_counts()


# Moving to tableau to make a heatmap visualization.
