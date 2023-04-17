
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# %%
"""
## Read datasets
"""

# %%
fake = pd.read_csv("D:/Project22-23/Fake_Tweet/Main123/Fake.csv/fake1.csv", encoding= 'unicode_escape')
true = pd.read_csv("D:/Project22-23/Fake_Tweet/Main123/True.csv/true1.csv", encoding= 'unicode_escape')

# %%
fake.shape

# %%
true.shape

# %%
"""
## Data cleaning and preparation
"""

# %%
# Add flag to track fake and real
fake['target'] = 'fake'
true['target'] = 'true'

# %%
# Concatenate dataframes
data = pd.concat([fake, true]).reset_index(drop = True)
data.shape

# %%
# Shuffle the data
from sklearn.utils import shuffle
data = shuffle(data)
data = data.reset_index(drop=True)

# %%
# Check the data
data.head()

# %%
# Removing the date (we won't use it for the analysis)
data.drop(["date"],axis=1,inplace=True)
data.head()

# %%
# Removing the title (we will only use the text)
data.drop(["title"],axis=1,inplace=True)
data.head()

# %%
# Convert to lowercase

data['text'] = data['text'].apply(lambda x: x.lower())
data.head()

# %%
# Remove punctuation

import string

def punctuation_removal(text):
    all_list = [char for char in text if char not in string.punctuation]
    clean_str = ''.join(all_list)
    return clean_str

data['text'] = data['text'].apply(punctuation_removal)

# %%
# Check
data.head()

# %%
# Removing stopwords
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english')

data['text'] = data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

# %%
data.head()

# %%
"""
## Basic data exploration
"""

# %%
# How many articles per subject?
print(data.groupby(['subject'])['text'].count())
data.groupby(['subject'])['text'].count().plot(kind="bar")
plt.show()

# %%
# How many fake and real articles?
print(data.groupby(['target'])['text'].count())
data.groupby(['target'])['text'].count().plot(kind="bar")
plt.show()