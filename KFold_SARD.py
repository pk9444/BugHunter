# K-Fold Model on SARD-Test-Suite Dataset

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import json
import pickle
from scipy.spatial import distance
import glob
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict


#---------------------------------Reading_Files-------------------------------------------------------------------#

# Read the patched (fixed) and vulnerable (flawed) code files in pandas dataframes
path_fixed_SARD_101 = r'D:\pycharm\projects\CapstoneMS\SARD_101_datasetFixes\149\**\*.c'
path_flawed_SARD_100 = r'D:\pycharm\projects\CapstoneMS\SARD_100_datasetFlaws\149\**\*.c'

# doc1 - patched/fixed, doc2 - vulnerable/flawed
doc1 = pd.concat((pd.read_csv(f,sep='delimiter',engine='python') for f in glob.iglob(path_fixed_SARD_101, recursive=True)), ignore_index=True)
doc2 = pd.concat((pd.read_csv(f,sep='delimiter',engine='python') for f in glob.iglob(path_flawed_SARD_100, recursive=True)), ignore_index=True)


#------------------------------ Pre_Processing--------------------------------------------------------------------#

def remove_comments(string):
    '''
    remove C-comments read as regex /*comments*/ and replace them with null
    :param string:
    :return: string (with comments removed
    '''
    string = re.sub(re.compile("/\*.*?\*/",re.DOTALL ) ,"" ,string) # remove all occurrences streamed comments (/*COMMENT */) from string
    string = re.sub(re.compile("//.*?\n" ) ,"" ,string) # remove all occurrence single-line comments (//COMMENT\n ) from string
    return string

# apply the function on doc1 and doc2
doc3 = remove_comments(doc1.to_string())
doc4 = remove_comments(doc2.to_string())

# define NLTK stopwords and a Porter Stemmer object for stemming
stop_words_list = set(stopwords.words("english"))
porter_Stemmer = PorterStemmer()


# method to apply preprocessing = stop word removal + stemming (using porter stemmer)
def apply_pre_processing(data_instance):
 '''

 :param data_instance: a dataframe in string readable format and comments striopped
 :return: stopwords eliminated, tokenized and porter stemmed dataframe in list format
 '''
 #sentences = sent_tokenize(data_instance.to_string().lower())
 #sentences = sent_tokenize(str(data_instance))
 sentences = sent_tokenize(data_instance)
 word_tokens = []
 for word in word_tokenize(str(sentences)):
     if word not in stop_words_list:
         word_tokens.append(word)

 porter_Stemmer.stem(str(word_tokens))

 processDataSet = json.dumps(word_tokens)
 processDataSet = [processDataSet]

 return processDataSet

# apply the function on doc3 and doc4 and preprocess them
pp_doc1 = apply_pre_processing(doc3)
pp_doc2 = apply_pre_processing(doc4)

#----------------------------------------Feature Engineering ------------------------------------------#

# Perform the TF-IDF Vectorization
tf_idf_vectorizer = TfidfVectorizer(ngram_range=(3,3)) #define the vectorizer object and set N_Grams
feature_vectors = tf_idf_vectorizer.fit_transform([str(pp_doc1), str(pp_doc2)]) # put them in a TF-IDF matrix
feature_names = tf_idf_vectorizer.get_feature_names() # get the feature names
dense = feature_vectors.todense()
denselist = dense.tolist()
dataframe = pd.DataFrame(denselist, columns=feature_names) # set the feature names as column names
print(dataframe)

# Perform TF-IDF Matrix Normalization
fix_df = dataframe.iloc[0].transpose()
fix_df2 = pd.DataFrame(fix_df)
fix_df2['Type'] = '0' # Set Class Label Type as 0 for patched feature vectors
#print(fix_df2)

flaw_df = dataframe.iloc[1].transpose()
flaw_df2 = pd.DataFrame(flaw_df)
flaw_df2['Type'] = '1' # Set Class Label Type as 1 for flawed feature vectors
#print(flaw_df2)

total_frames = fix_df2.append(flaw_df2) # Combine the dataframes into a whole training dataset
total_frames = total_frames.fillna(0) # Further clean the dataframe by getting rid of nan values
#print(total_frames)

# y = total_frames['Type']
# y = pd.DataFrame(y)
# print(y)
X = total_frames.drop('Type',axis=1) # Data to be trained
y = total_frames['Type'] # Target to which the data is to be predicted - Class Label: Type 0 or 1


# Scalar Transformation -
## A feature vector to be predicted at a time needs to be read as a scalar object (1-D vector)
scaler = StandardScaler() #
scaler.fit(X) #
X = scaler.transform(X) #

#---------------------------------------Training Phase -------------------------------------------------------#

# Define the Models and their specification
clf_GNB = GaussianNB()
clf_RF = RandomForestClassifier(n_estimators=100)
clf_LR = LogisticRegression(max_iter=1000)
clf_KNN = KNeighborsClassifier(n_neighbors=3)
clf_MLP = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1,max_iter=300)

# result1 = cross_val_score(clf_LR, X, y, cv=kf)
# result2 = cross_val_score(clf_GNB, X, y, cv=kf)

# Define the K for the cross validation
k = 3
kf = KFold(n_splits=k, random_state=None)
k1 = 10
kf1 = KFold(n_splits=k1, random_state=None)

# Prediction each model using k-value
y_pred_LR = cross_val_predict(clf_LR, X, y, cv=kf1)
cm_LR = confusion_matrix(y, y_pred_LR)

y_pred_GNB = cross_val_predict(clf_GNB, X, y, cv=kf1)
cm_GNB = confusion_matrix(y, y_pred_GNB)

y_pred_RF = cross_val_predict(clf_RF, X, y, cv=kf1)
cm_RF = confusion_matrix(y, y_pred_RF)

y_pred_KNN = cross_val_predict(clf_KNN, X, y, cv=kf1)
cm_KNN = confusion_matrix(y, y_pred_KNN)

y_pred_MLP = cross_val_predict(clf_MLP, X, y, cv=kf1)
cm_MLP = confusion_matrix(y, y_pred_MLP)

#print(result1)

#---------------------------------Model Evaluation ----------------------------------------------------#
print("\nConfusion Matrix : ")
print("----------------------------------------------------")
#print(result2.mean())

print("LR : " + str(cm_LR))
print("GNB : " + str(cm_GNB))
print("RF : " + str(cm_RF))
print("KNN : " + str(cm_KNN))
print("MLP : " + str(cm_MLP))

print("\nClassification Results : ")
print("----------------------------------------------------")

print("\nLogistic Regression : ")
classification_report_LR = classification_report(y, y_pred_LR)
print(classification_report_LR)

print("\nGaussian NB : ")
classification_report_GNB = classification_report(y, y_pred_GNB)
print(classification_report_GNB)

print("\nRandom Forest : ")
classification_report_RF = classification_report(y, y_pred_RF)
print(classification_report_RF)

print("\nKNN : ")
classification_report_KNN = classification_report(y, y_pred_KNN)
print(classification_report_KNN)

print("\nMultilayer Perceptron : ")
classification_report_MLP = classification_report(y, y_pred_MLP)
print(classification_report_MLP)


#--------------------------------------------Feature Extraction-------------------------------------#

print("\nCorrelation Metrics : ")
print("-------------------------------------------------------------------")
cosine_similarity = distance.cosine(dataframe.iloc[0],dataframe.iloc[1])
print("\ncosine_similarity : " + str(cosine_similarity))

eucledian_distance = distance.euclidean(dataframe.iloc[0],dataframe.iloc[1])
print("eucledian_distance : " + str(eucledian_distance))


def calc_KL_divergence(P,Q):
    output = 0
    DELTA = 0.00001
    P = P + DELTA
    Q = Q + DELTA
    for i in range(len(P)):
        output += P[i] * np.log(P[i]/Q[i])

    return output

fix_np = np.asarray(dataframe.iloc[0])
flaw_np = np.asarray(dataframe.iloc[1])

kl_divergence = calc_KL_divergence(fix_np,flaw_np)
print("KL-Divergence : " + str(kl_divergence))

print("-------------------------------------------------------------------")
# Extract 20 most common CVE keywords
print("Common flaw-associated keywords : ")
print(flaw_df.nlargest(20))

#-----------------------------------------Write to Production ------------------------------------------------#

#run these files and refactp
# clf_GNB.fit(X, y)
# pickle.dump(clf_GNB, open('KFold_GNB_SARD2G.pkl','wb'))
# clf_KNN.fit(X, y)
# pickle.dump(clf_KNN, open('KFold_KNN_SARD.pkl','wb'))
# clf_RF.fit(X, y)
# pickle.dump(clf_RF, open('KFold_RF_SARD.pkl','wb'))
# clf_LR.fit(X, y)
# pickle.dump(clf_LR, open('KFold_LR_SARD.pkl','wb'))
# clf_MLP.fit(X, y)
# pickle.dump(clf_MLP, open('KFold_MLP_SARD.pkl','wb'))

#--------------------------------------------------END---------------------------------------------------------#