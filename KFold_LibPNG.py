import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import json
import pickle
import joblib
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
from sklearn import decomposition, ensemble
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict

path_CVE_fixes_libpng = r'D:\pycharm\projects\Capstone\datasetsNew\libpng\CVE_PostfixFiles\**\**\*.c'
path_CVE_flaws_libpng = r'D:\pycharm\projects\Capstone\datasetsNew\libpng\CVE_PrefixFiles\**\**\*.c'

doc1 = pd.concat((pd.read_csv(f,sep='delimiter',engine='python') for f in glob.iglob(path_CVE_fixes_libpng, recursive=True)), ignore_index=True)
doc2 = pd.concat((pd.read_csv(f,sep='delimiter',engine='python') for f in glob.iglob(path_CVE_flaws_libpng, recursive=True)), ignore_index=True)

stop_words_list = set(stopwords.words("english"))
porter_Stemmer = PorterStemmer()

def remove_comments(string):
    string = re.sub(re.compile("/\*.*?\*/",re.DOTALL ) ,"" ,string) # remove all occurrences streamed comments (/*COMMENT */) from string
    string = re.sub(re.compile("//.*?\n" ) ,"" ,string) # remove all occurrence single-line comments (//COMMENT\n ) from string
    return string

doc3 = doc1.to_string().lower()
doc4 = doc2.to_string().lower()

doc5 = remove_comments(doc3)
doc6 = remove_comments(doc4)

#print(doc5)

sentences1 = sent_tokenize(doc5)
# sentences = sent_tokenize(str(data_instance))
word_tokens1 = []
for word1 in word_tokenize(str(sentences1)):
    if word1 not in stop_words_list:
        word_tokens1.append(word1)

porter_Stemmer.stem(str(word_tokens1))

processDataSet1 = json.dumps(word_tokens1)
processDataSet1 = [processDataSet1]

sentences2 = sent_tokenize(doc6)
# sentences = sent_tokenize(str(data_instance))
word_tokens2 = []
for word2 in word_tokenize(str(sentences2)):
    if word2 not in stop_words_list:
        word_tokens2.append(word2)

porter_Stemmer.stem(str(word_tokens2))

processDataSet2 = json.dumps(word_tokens2)
processDataSet2 = [processDataSet2]

tf_idf_vectorizer = TfidfVectorizer(ngram_range=(3,3))
feature_vectors = tf_idf_vectorizer.fit_transform([str(processDataSet1), str(processDataSet2)])
feature_names = tf_idf_vectorizer.get_feature_names()
dense = feature_vectors.todense()
denselist = dense.tolist()
dataframe = pd.DataFrame(denselist, columns=feature_names)
print(dataframe)

# post-processing phase
fix_df = dataframe.iloc[0].transpose()
fix_df2 = pd.DataFrame(fix_df)
fix_df2['Type'] = '0'
#print(fix_df2)

flaw_df = dataframe.iloc[1].transpose()
flaw_df2 = pd.DataFrame(flaw_df)
flaw_df2['Type'] = '1'
#print(flaw_df2)

total_frames = fix_df2.append(flaw_df2)
total_frames = total_frames.fillna(0)
print(total_frames)

# y = total_frames['Type']
# y = pd.DataFrame(y)
# print(y)
X = total_frames.drop('Type',axis=1)
y = total_frames['Type']
#print(X)
#print(y)

scaler = StandardScaler() #
scaler.fit(X) #
X = scaler.transform(X) #

clf_GNB = GaussianNB()
clf_RF = RandomForestClassifier(n_estimators=100)
clf_LR = LogisticRegression(max_iter=1000)
clf_KNN = KNeighborsClassifier(n_neighbors=3)
clf_MLP = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1,max_iter=300)

# result1 = cross_val_score(clf_LR, X, y, cv=kf)
# result2 = cross_val_score(clf_GNB, X, y, cv=kf)
k = 3
kf = KFold(n_splits=k, random_state=None)

y_pred_LR = cross_val_predict(clf_LR, X, y, cv=kf)
cm_LR = confusion_matrix(y, y_pred_LR)

y_pred_GNB = cross_val_predict(clf_GNB, X, y, cv=kf)
cm_GNB = confusion_matrix(y, y_pred_GNB)

y_pred_RF = cross_val_predict(clf_RF, X, y, cv=kf)
cm_RF = confusion_matrix(y, y_pred_RF)

y_pred_KNN = cross_val_predict(clf_KNN, X, y, cv=kf)
cm_KNN = confusion_matrix(y, y_pred_KNN)

y_pred_MLP = cross_val_predict(clf_MLP, X, y, cv=kf)
cm_MLP = confusion_matrix(y, y_pred_MLP)

#print(result1)
print("\nConfusion Matrix : ")
print("----------------------------------------------------")

#print(result2.mean())

print("LR : " + "\n" + str(cm_LR))
print("\nGNB : " + "\n" + str(cm_GNB))
print("\nRF : " + "\n" + str(cm_RF))
print("\nKNN : " + "\n" + str(cm_KNN))
print("\nMLP : " + "\n" + str(cm_MLP))


print("\nClassification Reports : ")
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

print("\nFeature Engineering Results : ")
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

print("\nCommon-Keywords : ")
print(flaw_df.nlargest(10))
