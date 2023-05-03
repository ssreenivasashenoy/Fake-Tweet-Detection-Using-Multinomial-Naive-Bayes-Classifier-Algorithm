# These lines import the necessary libraries, pandas and numpy, which are used for data manipulation and numerical computing respectively.
import pandas as pd
import numpy as np


# These lines read in two csv files, one containing fake news data and the other containing true news data, and store them in pandas dataframes.
df_fake = pd.read_csv("D:/FinalProject/project/Fake.csv/Fake.csv", encoding= 'unicode_escape')
df_true = pd.read_csv("D:/FinalProject/project/True.csv/true12.csv", encoding= 'unicode_escape')


#This line displays the first five rows of the df_fake dataframe.
df_fake.head()


# These lines take a random sample of 1000 rows from each dataframe and overwrite the original dataframes with the new samples.
df_fake = df_fake.sample(1000)
df_true = df_true.sample(1000)


#This line displays the number of rows and columns in each dataframe after the samples have been taken.
df_fake.shape , df_true.shape   


# These lines add a new column to each dataframe called "nature" and assign the value 0 to df_fake and 1 to df_true.
df_fake['nature'] = 0
df_true['nature'] = 1


# This line displays the number of rows and columns in each dataframe after the "nature" column has been added.
df_fake.shape , df_true.shape    


# These two lines of code remove the columns titled 'title', 'subject', and 'date' from the dataframes df_fake and df_true in-place, meaning the dataframes are modified directly without creating a copy.
df_fake.drop(['title', 'subject', 'date'], axis=1, inplace=True)
df_true.drop(['title', 'subject', 'date'], axis=1, inplace=True)


# This line of code concatenates the dataframes df_fake and df_true vertically (i.e., along axis 0) to create a single dataframe df. The ignore_index argument is set to True to reset the index of the concatenated dataframe.
df = pd.concat([df_fake, df_true], axis=0, ignore_index=True)


# These two lines of code create two variables X and y, which are used for training and testing a machine learning model.
X = df.text		# X contains the 'text' column of the df dataframe
y = df.nature	# y contains the 'nature' column, which indicates whether an article is real or fake news.


from sklearn.model_selection import train_test_split	# This code imports the train_test_split function from the model_selection module of Scikit-learn.


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, stratify=y, random_state=1)


# The train_test_split function is then used to split the data X and the corresponding labels y into training and testing sets, with a test size of 33% (test_size=0.33).
# The stratify parameter is used to ensure that the proportion of the target variable y is the same in both the training and testing sets, which is helpful when dealing with imbalanced data.
# The random_state parameter is set to 1 to ensure reproducibility of the results.


from sklearn.feature_extraction.text import CountVectorizer	# This code imports the CountVectorizer class from the feature_extraction.text module of Scikit-learn.


cv1 = CountVectorizer(ngram_range=(1,1))
cv2 = CountVectorizer(ngram_range=(1,2))
# Two instances of the CountVectorizer class are created, one for unigrams (cv1) and the other for bigrams (cv2), which will be used to convert the text data into feature vectors.


X_train_vector1 = cv1.fit_transform(X_train)		# X_train_vector1 contains the unigram feature vectors
X_train_vector2 = cv2.fit_transform(X_train)		# X_train_vector2 contains the bigram feature vectors.


pd.Series(list(cv1.vocabulary_.items())).sample(5)
pd.Series(list(cv2.vocabulary_.items())).sample(5)
# These lines create two Pandas Series that contain five randomly sampled items each from the vocabulary dictionary of the unigram and bigram CountVectorizer objects, respectively.
# The vocabulary_ attribute of the CountVectorizer class is a dictionary that maps each token (unigram/bigram) to a unique integer index.


# These lines convert the unigram and bigram feature vectors stored in X_train_vector1 and X_train_vector2 into NumPy arrays using the toarray() method of the scipy.sparse matrix.
X_train_array1 = X_train_vector1.toarray()
X_train_array2 = X_train_vector2.toarray()
# This is necessary because many ML in Scikit-learn expect input data to be in the form of NumPy arrays.


# These lines transform the text data in the X_test array into feature vectors using the transform() method of the CountVectorizer class.
X_test_vector1= cv1.transform(X_test)	# X_test_vector1 contains the unigram feature vectors 
X_test_vector2= cv2.transform(X_test)	# X_test_vector2 contains the bigram feature vectors.


X_test_array1 = X_test_vector1.toarray()   # Converting sparse matrix to dense matrix for X_test_vector1
X_test_array2 = X_test_vector2.toarray()   # Converting sparse matrix to dense matrix for X_test_vector2


from sklearn.naive_bayes import MultinomialNB    # Importing Multinomial Naive Bayes Classifier from Scikit-Learn
model = MultinomialNB()    # Creating an instance of MultinomialNB classifier


model.fit(X_train_array1,y_train)    # Training the model using X_train_array1 and y_train
score1 = model.score(X_test_array1, y_test )    # Testing the model on X_test_array1 and y_test, and calculating the accuracy
score1   # Printing the accuracy of the model for X_test_array1


y_pred1 = model.predict(X_test_array1)    # Making predictions on X_test_array1 using the trained model


model.fit(X_train_array2,y_train)    # Training the model using X_train_array2 and y_train
score2 = model.score(X_test_array2, y_test )    # Testing the model on X_test_array2 and y_test, and calculating the accuracy
score2   # Printing the accuracy of the model for X_test_array2


y_pred2 = model.predict(X_test_array2)    # Making predictions on X_test_array2 using the trained model


score = {'Name' : ['Only_Unigram' , 'Uni_&_Bigram' ],    # Creating a dictionary containing the model names and their accuracies
      'Score' : [score1, score2 ]}
score_df = pd.DataFrame(score)   # Creating a dataframe from the dictionary


import matplotlib.pyplot as plt    # Importing matplotlib library for plotting


from sklearn.metrics import classification_report, confusion_matrix   # Importing classification_report and confusion_matrix functions from Scikit-Learn


import matplotlib.pyplot as plt   # Importing matplotlib library for plotting
import seaborn as sn   # Importing seaborn library for creating heatmaps


print(f'Classification Report of Only Unigram :\n\n{classification_report(y_test , y_pred1)} \n\n\n'  # Printing the classification report for the model trained on X_train_array1
      f'Classification Report of Uni and Bigram :\n\n{classification_report(y_test , y_pred2)}\n\n\n'   # Printing the classification report for the model trained on X_train_array2
)


# Import necessary libraries
import seaborn as sn
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


# Assuming y_test and y_pred1, y_pred2 are previously defined arrays of true and predicted labels


# Generate confusion matrices using scikit-learn's confusion_matrix function
cm1 = confusion_matrix(y_test, y_pred1)
cm2 = confusion_matrix(y_test, y_pred2)


# Print the confusion matrices
print("cm1 : \n", cm1 ,"\n\n","cm2 : \n", cm2 , "\n\n")


# Plot heatmap of confusion matrix for model with only unigrams
plt.figure(figsize=(2,2))
sn.heatmap(cm1 , annot=True , fmt='d') # annot=True to show the values within each cell and fmt='d' to format as integer
plt.xlabel("Truth")
plt.ylabel("Predicted")
plt.title("Only Unigram")
plt.show()


# Plot heatmap of confusion matrix for model with both unigrams and bigrams
plt.figure(figsize=(2,2))
sn.heatmap(cm2 , annot=True , fmt='d') # annot=True to show the values within each cell and fmt='d' to format as integer
plt.xlabel("Truth")
plt.ylabel("Predicted")
plt.title("Uni and Bigram")
plt.show()