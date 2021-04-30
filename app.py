# Deploy to Production as a web service

from flask import Flask, render_template, request, redirect, url_for
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import pickle
from joblib import load
import numpy as np
from sklearn.preprocessing import StandardScaler

# Read the pickle file where the ML-model is dumped

GNB_KFold_SARD = pickle.load(open('KFold_GNB_SARD.pkl','rb'))
GNB_KFold_SARD2G = pickle.load(open('KFold_GNB_SARD2G.pkl','rb'))
GNB_KFold_Linux = pickle.load(open('KFold_GNB_Linux.pkl','rb'))

# define the flask object which will run on the server and parse the UI
app = Flask(__name__)

# define NLTK stopwords and a Porter Stemmer object for stemming
stop_words_list = set(stopwords.words("english"))
porter_Stemmer = PorterStemmer()

# Route the home page of the web app
@app.route('/')
def index():
    return render_template('index.html')

# Route the page where ML-model performs prediction and generates results
@app.route('/detection', methods=['POST','GET'])
def detect():
    """
    perform the vulnerability detection as per the trained Machine Learning Model from the pickle file
    :return: XMP format output containing all the predictions of the feature vectors in the /detection route
    """
    data = request.form['subject'] #read data from the text area - convert it to be suitable for the ML-pipeline
    #df = pandas.DataFrame([data])
    df = pd.DataFrame([data]) # convert it to a pandas dataframe
    sentences = sent_tokenize(df.to_string().lower()) # perform tokenization as readable dataframe string
    # sentences = sent_tokenize(str(data_instance))
    word_tokens = [] # tokens that will contain all our tokenized words from sentences
    for word in word_tokenize(str(sentences)):
        if word not in stop_words_list:
            word_tokens.append(word)

    porter_Stemmer.stem(str(word_tokens)) # porter stem the word tokens

    processDataSet = json.dumps(word_tokens)
    processDataSet = [processDataSet]  # read into a JSON array

    # Perform the TF-IDF Vectorization
    tf_idf_vectorizer = TfidfVectorizer()
    feature_vectors = tf_idf_vectorizer.fit_transform(processDataSet)
    feature_names = tf_idf_vectorizer.get_feature_names()
    dense = feature_vectors.todense()
    denselist = dense.tolist()
    dataframe = pd.DataFrame(denselist, columns=feature_names)
    df_T = dataframe.iloc[0].transpose()
    df_N = pd.DataFrame(df_T)
    df_N_temp = pd.DataFrame(df_T)

    scaler = StandardScaler()  #
    scaler.fit(df_N)  #
    df_N = scaler.transform(df_N)  #

    prediction = GNB_KFold_SARD.predict(df_N)

    df_N_arr = []
    df_N_arr.append(df_N_temp)
    #--------------------------#

    df_N_temp['Prediction'] = prediction

    return "<xmp>"  "\n" + str(df_N_temp) + "\n" + " </xmp> " #

# Driver function
if __name__ == '__main__':
    app.run(debug=True)

#----------------------------------------------------END--------------------------------------------------------#