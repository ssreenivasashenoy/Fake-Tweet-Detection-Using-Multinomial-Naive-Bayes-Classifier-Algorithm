<div align="center">
    <img src="https://github.com/ssreenivasashenoy/Fake-Tweet-Detection-Using-Multinomial-Naive-Bayes-Classifier-Algorithm/blob/main/Others/TeamLogo.png" width="200" height="200">
</div>


# Fake Tweet Detection Using Multinomial Naive Bayes Classifier Algorithm

"Fake Tweet Detection Using Multinomial Naive Bayes Classifier Algorithm" is a project that aims to develop a machine learning model to identify fake tweets on social media platforms such as Twitter. The project utilizes the Multinomial Naive Bayes Classifier algorithm, which is a probabilistic algorithm commonly used for text classification tasks.

The model is trained on a dataset of tweets labeled as either real or fake. The dataset is preprocessed by removing stop words, stemming, and tokenization to transform the text data into a format that can be used for modeling. The Multinomial Naive Bayes Classifier algorithm then uses these features to calculate the probability of a tweet being real or fake.

The project aims to achieve high accuracy in identifying fake tweets, which can help reduce the spread of false information on social media platforms. The model can also be further improved by incorporating other machine learning algorithms or using more advanced natural language processing techniques.


## Dataset

The dataset used for this project is a collection of tweets labeled as either real or fake. The dataset is split into two parts: a training set and a test set. The training set is used to train the machine learning model, while the test set is used to evaluate the performance of the model. The dataset is taken from [Kaggle](https://www.kaggle.com).


## Preprocessing

The first step in the project is to preprocess the data. Before feeding the data into the model, we need to preprocess it. This includes removing special characters, URLs, and stop words, and converting all letters to lowercase. We will be using the NLTK library for this purpose.


## Feature Extraction

The next step is to extract features from the preprocessed text. In this project, we used the Term Frequency-Inverse Document Frequency (TF-IDF) vectorization method to convert the text into numerical features.


## Training the Model

After preprocessing and feature extraction, we can now train the Multinomial Naive Bayes classifier algorithm. We will be using the scikit-learn library for this purpose.


## Multinomial Naive Bayes Classifier Algorithm

The Multinomial Naive Bayes (MNB) classifier algorithm is a probabilistic algorithm used for classification problems. It is based on Bayes' theorem and assumes that the features are conditionally independent given the class label. The MNB algorithm is particularly well-suited for text classification tasks, such as fake tweet detection.


## System Architecture Diagram

<img src="https://github.com/ssreenivasashenoy/Fake-Tweet-Detection-Using-Multinomial-Naive-Bayes-Classifier-Algorithm/blob/main/Others/SystemArchitechture.png" width="600" height="300">


## Prerequisites

The following libraries are required to run the project:
- Python 3
- scikit-learn
- pandas
- numpy
- matplotlib


## Getting Started

To get started, 
1. Clone the repository:
```cmd
    git clone https://github.com/ssreenivasashenoy/Fake-Tweet-Detection-Using-Multinomial-Naive-Bayes-Classifier-Algorithm.git
```

2. Navigate to the project directory:
```cmd
    cd Fake-Tweet-Detection-Using-Multinomial-Naive-Bayes-Classifier-Algorithm
```

3. Install the required libraries:
```cmd
    pip install -r requirements.txt
```

4. Open the `app.py` file and modify the file paths to point to the location of the real and fake news CSV files on your local machine.

5. Run the `app.py` file:
```cmd
    python app.py
```


## Screenshots

### Landing Page
<img src="https://github.com/ssreenivasashenoy/Fake-Tweet-Detection-Using-Multinomial-Naive-Bayes-Classifier-Algorithm/blob/main/Others/landingpage.jpeg" width="600" height="300">

### Output Page

#### Fake
<img src="https://github.com/ssreenivasashenoy/Fake-Tweet-Detection-Using-Multinomial-Naive-Bayes-Classifier-Algorithm/blob/main/Others/predictfake.jpeg" width="600" height="300">

#### Real
<img src="https://github.com/ssreenivasashenoy/Fake-Tweet-Detection-Using-Multinomial-Naive-Bayes-Classifier-Algorithm/blob/main/Others/predicttrue.jpeg" width="600" height="300">


## Evaluation

The performance of the model is evaluated using the accuracy, precision, recall, and F1 score metrics. The testing dataset is used to evaluate the performance of the model.


## Conclusion

Fake news detection is an important problem in today's world, and machine learning can be used to automate the process. The Multinomial Naive Bayes Classifier Algorithm is a simple yet effective algorithm for this task.


## Project Poster

<div align="center">
<img src="https://github.com/ssreenivasashenoy/Fake-Tweet-Detection-Using-Multinomial-Naive-Bayes-Classifier-Algorithm/blob/main/Others/ProjectPoster.png" width="400" height="600">
</div>


## Team members

- S Sreenivasa Shenoy [@ssreenivasashenoy](https://github.com/ssreenivasashenoy)
- P Padmaprasad Shenoy [@ppadmaprasadshenoy](https://github.com/ppadmaprasadshenoy)
- Subramanya A Shet [@believer-20](https://github.com/believer-20)
- Suhas S Kamath [@suhas-kamath](https://github.com/suhas-kamath)
