from flask import Flask, render_template,request,session,flash		# Imports the Flask framework and other necessary libraries
import sqlite3 as sql												# Imports the sqlite3 module for working with SQLite databases.
import os															# Imports the os module, which provides a way of interacting with the operating system.
from csv import writer												# Imports the writer function from the csv module, which will be used to write data to a CSV file.
from sklearn.model_selection import train_test_split				# Imports the train_test_split function from the model_selection module of Scikit-learn, which will be used to split the data into training and testing sets.
from sklearn.feature_extraction.text import TfidfVectorizer			# Imports the TfidfVectorizer class from the feature_extraction.text module of Scikit-learn, which will be used to convert text data into numerical vectors.
from sklearn.linear_model import PassiveAggressiveClassifier		# Imports the PassiveAggressiveClassifier class from the linear_model module of Scikit-learn, which will be used to train a machine learning model to classify news articles as either real or fake.
from sklearn.metrics import accuracy_score, confusion_matrix		# Imports the accuracy_score and confusion_matrix functions from the metrics module of Scikit-learn, which will be used to evaluate the performance of the machine learning model.
import pandas as pd													# Imports the Pandas library, which will be used to manipulate data.
import pickle														# Imports the pickle module, which will be used to save and load the machine learning model to and from a file.


app = Flask(__name__)												# Creates a new instance of the Flask class, which will be used to define the application and its routes.
# model = pickle.load(open('pac.pkl', 'rb'))
# tfidf = pickle.load(open('tfidf_vectorizer.pkl','rb'))


@app.route('/')														# This line defines the home page route of the application, which is the root URL ("/").
def home():															# This function is executed when a user visits the home page of the application.
	return render_template('home.html')								# This function is used to render a HTML template named home.html which will be displayed to the user.


@app.route('/predict',methods=['POST'])								# Defines a Flask route to handle incoming POST requests
def predict():														# Defines a function to handle the request
		
	if request.method == 'POST':									# Check if the request method is POST
		news = request.form['news']									# Get the 'news' data from the POST request form
		#data = [news]
		#lowercase = ques.lower()
		#print(lowercase)

		dataframe = pd.read_csv('true1234.csv')						# Load a CSV file into a pandas dataframe
		dataframe.head()											# Display the first 5 rows of the DataFrame to quickly check that the data has been loaded correctly.

		# %%
		x = dataframe['text']
		y = dataframe['label']
		# Extract the 'text' and 'label' columns from the dataframe into x and y respectively

		# Import required modules from scikit-learn for machine learning tasks
		from sklearn.model_selection import train_test_split
		from sklearn.feature_extraction.text import TfidfVectorizer
		from sklearn.linear_model import PassiveAggressiveClassifier
		from sklearn.metrics import accuracy_score, confusion_matrix

		# %%
		x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)	# Split the data into training and testing sets
		y_train

		# %%
		y_train

		# %%
		tfvect = TfidfVectorizer(stop_words='english',max_df=0.7)			# Initialize a TfidfVectorizer to transform the text data into feature vectors
		tfid_x_train = tfvect.fit_transform(x_train)						# Fit and transform the training set
		tfid_x_test = tfvect.transform(x_test)								# Transform the testing set using the same vectorizer

		# %%
		"""
		* max_df = 0.50 means "ignore terms that appear in more than 50% of the documents".
		* max_df = 25 means "ignore terms that appear in more than 25 documents".
		"""

		# %%
		# Initialize a PassiveAggressiveClassifier and fit it to the training data
		classifier = PassiveAggressiveClassifier(max_iter=50)
		classifier.fit(tfid_x_train,y_train)

		# %%
		# Predict on the testing data and calculate the accuracy score
		y_pred = classifier.predict(tfid_x_test)
		score = accuracy_score(y_test,y_pred)
		print('Accuracy:', score*100)
		#print('Accuracy: {round(score*100,2)}%')

		# %%
		# Calculate the confusion matrix
		cf = confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])
		print(cf)

		# %%
		def fake_news_det(news):												# Define a function to detect fake news
			input_data = [news]													# Takes 'news' data as input
			print('input_data',input_data)										
			vectorized_input_data = tfvect.transform(input_data)				# Vectorizes the input using the same TfidfVectorizer used for training
			prediction = classifier.predict(vectorized_input_data)				# Predicts the label using the trained PassiveAggressiveClassifier
			print(prediction)
			return prediction													# Returns the predicted label

		# %%
		response = fake_news_det(news)

		#vect = tfidf.transform(data)
		my_prediction = fake_news_det(news)										# Call the fake_news_det function on the 'news' data and store the predicted label

		# Render the prediction.html template with the predicted label as input for display
		return render_template('prediction.html', prediction=my_prediction)

# Checks whether the current module is being run as the main program. 
# The code inside this block will only be executed if this condition is True.
if __name__ == '__main__':
    app.run(debug=True)

# The line inside 'if' starts the Flask application, which is stored in the app variable. 
# The run method starts the development server and listens for incoming requests. 
# The debug=True argument tells Flask to run in debug mode, which enables additional error messages and other helpful features during development.