# Import necessary libraries
import pandas as pd                                                      # library for data manipulation and analysis
import numpy as np                                                       # library for numerical operations
import matplotlib.pyplot as plt                                          # library for plotting graphs and charts
import seaborn as sns                                                    # library for data visualization
from sklearn.feature_extraction.text import CountVectorizer              # library for text preprocessing
from sklearn.feature_extraction.text import TfidfTransformer             # library for text preprocessing
from sklearn import feature_extraction, linear_model, model_selection, preprocessing  # libraries for machine learning
from sklearn.metrics import accuracy_score                               # library for evaluation metric
from sklearn.model_selection import train_test_split                     # library for data splitting
from sklearn.pipeline import Pipeline                                    # library for creating data processing pipelines

# Read in the datasets
fake = pd.read_csv("D:/FinalProject/project/Fake.csv/Fake.csv", encoding= 'unicode_escape')
true = pd.read_csv("D:/FinalProject/project/True.csv/true12.csv", encoding= 'unicode_escape')

# Print the shape of the datasets (rows and columns)
print(fake.shape)
print(true.shape)

# Add a column 'target' to track fake and real news in the datasets
fake['target'] = 'fake'
true['target'] = 'true'

# Concatenate the datasets
data = pd.concat([fake, true]).reset_index(drop = True)

# Shuffle the data to avoid any bias
from sklearn.utils import shuffle
data = shuffle(data)

# Reset the index of the shuffled dataset
data = data.reset_index(drop=True)

# Remove the 'date' and 'title' columns from the dataset as they are not needed for analysis
data.drop(["date"],axis=1,inplace=True)
data.drop(["title"],axis=1,inplace=True)

# Convert all text to lowercase to avoid case sensitivity
data['text'] = data['text'].apply(lambda x: x.lower())

import string           # Importing the 'string' module which contains various string constants and functions.

# This punctuation_removal function removes all the punctuations from the text and returns the cleaned string.
def punctuation_removal(text):
    all_list = [char for char in text if char not in string.punctuation]
    clean_str = ''.join(all_list)
    return clean_str

# Applies the 'punctuation_removal' function to the 'text' column of the 'data' dataframe.
data['text'] = data['text'].apply(punctuation_removal)

# Importing the Natural Language Toolkit (nltk) library and downloading the 'stopwords' package which contains a list of commonly used English stop words.
import nltk
nltk.download('stopwords')

# Importing the 'stopwords' package from the nltk.corpus library and assigning the English stop words to a variable 'stop'.
from nltk.corpus import stopwords
stop = stopwords.words('english')

# Applies a lambda function to the 'text' column of the 'data' dataframe, which splits each text into a list of words, 
# removes all stop words from the list and joins the remaining words back into a sentence. 
data['text'] = data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

# Prints and displays a bar plot of the count of 'text' values grouped by 'subject' in the 'data' dataframe.
print(data.groupby(['subject'])['text'].count())
data.groupby(['subject'])['text'].count().plot(kind="bar")
plt.show()

# Prints and displays a bar plot of the count of 'text' values grouped by 'target' in the 'data' dataframe.
print(data.groupby(['target'])['text'].count())
data.groupby(['target'])['text'].count().plot(kind="bar")
plt.show()
