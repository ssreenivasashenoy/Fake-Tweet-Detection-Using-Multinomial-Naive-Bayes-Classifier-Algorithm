# %%
import pandas as pd
import numpy as np

# %%
"""
## Reading csv
"""

# %%
df_fake = pd.read_csv("D:/Project22-23/Fake_Tweet/Main123/Fake.csv/fake1.csv", encoding= 'unicode_escape')
df_true = pd.read_csv("D:/Project22-23/Fake_Tweet/Main123/True.csv/true1.csv", encoding= 'unicode_escape')

# %%
"""
## Taking a sample of 1000 only. 
### (My CPU crashed everytime i took the full data...haha)
"""

# %%
df_fake.head()

df_fake = df_fake.sample(1000)
df_true = df_true.sample(1000)

# %%
df_fake.shape  , df_true.shape

# %%
"""
## Assigning "nature" to news type
"""

# %%
df_fake['nature'] = 0
df_true['nature'] = 1

df_fake.shape , df_true.shape

# %%
"""
## Concating the two DataFrame
"""

# %%
df_fake.drop(['title' , 'subject', 'date'], axis = 1 , inplace= True)

df_true.drop(['title' , 'subject', 'date'], axis = 1 , inplace= True)

df = pd.concat([df_fake, df_true], axis = 0, ignore_index= True )

df.sample(5)

# %%
"""
## Selecting X and y
"""

# %%
X = df.text
y = df.nature

# %%
"""
## Splitting the Data
"""

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y , test_size=0.33 , stratify= y  , random_state =1)

# %%
"""
## Vectorizing the text data into four categories: 
### UNIGRAM ONLY
### UNI & BIGRAM
### UNI, BI AND TRIGRAM
### BIGRAM ONLY
"""

# %%
from sklearn.feature_extraction.text import CountVectorizer

cv1 = CountVectorizer(ngram_range=(1,1))
cv2 = CountVectorizer(ngram_range=(1,2))




# %%
X_train_vector1 = cv1.fit_transform(X_train)
X_train_vector2 = cv2.fit_transform(X_train)


# %%
"""
## VOCABULARY of four categories
"""

# %%
pd.Series(list(cv1.vocabulary_.items())).sample(5)

# %%
pd.Series(list(cv2.vocabulary_.items())).sample(5)


# %%
"""
## Vector TO Array
"""

# %%
X_train_array1 = X_train_vector1.toarray()
X_train_array2 = X_train_vector2.toarray()


# %%
"""
## Same thing with the Test Data
"""

# %%
X_test_vector1= cv1.transform(X_test)
X_test_vector2= cv2.transform(X_test)


# %%
X_test_array1 = X_test_vector1.toarray()
X_test_array2 = X_test_vector2.toarray()


# %%
"""
## MODEL TIME with all four categories
"""

# %%
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()

# %%
model.fit(X_train_array1,y_train)
score1 = model.score(X_test_array1, y_test )
score1

# %%
y_pred1 = model.predict(X_test_array1)

# %%
model.fit(X_train_array2,y_train)
score2 = model.score(X_test_array2, y_test )
score2

# %%
y_pred2 = model.predict(X_test_array2)
'''
# %%
model.fit(X_train_array3,y_train)
score3 = model.score(X_test_array3, y_test )
score3

# %%
y_pred3 = model.predict(X_test_array3)

# %%
model.fit(X_train_array4,y_train)
score4 = model.score(X_test_array4, y_test )
score4

# %%
y_pred4 = model.predict(X_test_array4)
'''
# %%
"""
## Making the scores visualy more attractive
"""

# %%
score = {'Name' : ['Only_Unigram' , 'Uni_&_Bigram' ],
         'Score' : [score1, score2 ]}
score_df = pd.DataFrame(score)
score_df

# %%
"""
## Best Part of the Entire Notebook : Colorful it is !
"""

# %%
import matplotlib.pyplot as plt

#plt.figure(figsize=(10,5),facecolor= 'pink', edgecolor= 'black')





# %%
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt

import seaborn as sn


# %%
"""
## Classification Report (For Nerds)
"""

# %%
print(f'Classification Report of Only Unigram :\n\n{classification_report(y_test , y_pred1)} \n\n\n'
      f'Classification Report of Uni and Bigram :\n\n{classification_report(y_test , y_pred2)}\n\n\n'
      )

# %%
"""
## To kill the confusion , here is the MATRIX
"""

# %%
cm1 = confusion_matrix(y_test , y_pred1)
cm2 = confusion_matrix(y_test, y_pred2)


print("cm1 : \n", cm1 ,"\n\n","cm2 : \n", cm2 , "\n\n")

# %%
"""
## OOPS ! this increased the confusion
"""

# %%
"""
## LETS ADD COLORS AND TRY AGAIN WITH HEATMAP
"""

# %%
plt.figure(figsize=(2,2))

sn.heatmap(cm1 , annot =True , fmt = 'd')
plt.xlabel("Truth")
plt.ylabel("Predicted")
plt.title("Only Unigram")
plt.show()


# %%
plt.figure(figsize=(2,2))

sn.heatmap(cm2 , annot =True , fmt = 'd')
plt.xlabel("Truth")
plt.ylabel("Predicted")
plt.title("Uni and Bigram")
plt.show()


