from flask import Flask, render_template,request,session,flash
import sqlite3 as sql
import os
from csv import writer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import pickle

app = Flask(__name__)
# model = pickle.load(open('pac.pkl', 'rb'))
# tfidf = pickle.load(open('tfidf_vectorizer.pkl','rb'))

@app.route('/')
def home():
	return render_template('home.html')


@app.route('/predict',methods=['POST'])
def predict():
		
	if request.method == 'POST':
		news = request.form['news']
		#data = [news]
		#lowercase = ques.lower()
		#print(lowercase)

		dataframe = pd.read_csv('true1234.csv')
		dataframe.head()

		# %%
		x = dataframe['text']
		y = dataframe['label']

		from sklearn.model_selection import train_test_split
		from sklearn.feature_extraction.text import TfidfVectorizer
		from sklearn.linear_model import PassiveAggressiveClassifier
		from sklearn.metrics import accuracy_score, confusion_matrix

		# %%
		x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
		y_train

		# %%
		y_train

		# %%
		tfvect = TfidfVectorizer(stop_words='english',max_df=0.7)
		tfid_x_train = tfvect.fit_transform(x_train)
		tfid_x_test = tfvect.transform(x_test)

		# %%
		"""
		* max_df = 0.50 means "ignore terms that appear in more than 50% of the documents".
		* max_df = 25 means "ignore terms that appear in more than 25 documents".
		"""

		# %%
		classifier = PassiveAggressiveClassifier(max_iter=50)
		classifier.fit(tfid_x_train,y_train)

		# %%
		y_pred = classifier.predict(tfid_x_test)
		score = accuracy_score(y_test,y_pred)
		print('Accuracy:', score*100)
		#print('Accuracy: {round(score*100,2)}%')

		# %%
		cf = confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])
		print(cf)

		# %%
		def fake_news_det(news):
			input_data = [news]
			print('input_data',input_data)
			vectorized_input_data = tfvect.transform(input_data)
			prediction = classifier.predict(vectorized_input_data)
			print(prediction)
			return prediction

		# %%
		response = fake_news_det(news)


		#vect = tfidf.transform(data)
		my_prediction = fake_news_det(news)




		return render_template('prediction.html', prediction=my_prediction)

if __name__ == '__main__':
    app.run(debug=True)
