#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[6]:


comments = pd.read_csv(r'C:\Users\azade\OneDrive\Desktop\youtube/UScomments.csv', error_bad_lines=False)


# In[7]:


comments.head()


# In[9]:


comments.isnull()


# In[10]:


comments.isnull().sum()


# In[11]:


comments.dropna(inplace=True)


# In[12]:


comments.isnull().sum()


# In[13]:


get_ipython().system('pip install textblob')


# In[15]:


from textblob import TextBlob


# In[16]:


comments.head(6)


# In[18]:


TextBlob("Logan Paul it's yo big day ‼️‼️‼️").sentiment.polarity


# In[20]:


comments.shape


# In[22]:


sample_df=comments[0:1000]
sample_df.shape


# In[ ]:





# In[28]:


polarity = []
for comment in comments['comment_text']:
    try:
        polarity.append(TextBlob(comment).sentiment.polarity)
    except:
        polarity.append(0)


# In[29]:


len(polarity)


# In[30]:


comments['polarity']=polarity


# In[32]:


comments.head(20)


# In[ ]:




