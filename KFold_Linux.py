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

path_CVE_fixes_linux = r'D:\pycharm\projects\Capstone\datasetsNew\linux2\CVE_PostfixFiles\**\**\*.c'
path_CVE_flaws_linux = r'D:\pycharm\projects\Capstone\datasetsNew\linux2\CVE_PrefixFiles\**\**\*.c'
path1 = r'D:\pycharm\projects\Capstone\datasetsNew\linux2\CVE_flaw1\**\**\*.c'
path2 = r'D:\pycharm\projects\Capstone\datasetsNew\linux2\CVE_fix1\**\**\*.c'



stop_words_list = set(stopwords.words("english"))
porter_Stemmer = PorterStemmer()

print("-------------------------------------------------------")

doc1 = []
doc2 = []

# ---------------------------------------------------------------------------
# for file in glob.iglob(path_CVE_fixes_linux):
#     with open(file, errors="ignore") as f:
#         contents_1 = f.read()
#         doc1.append(contents_1)
#
#
# for file in glob.iglob(path_CVE_flaws_linux):
#     with open(file, errors="ignore") as f:
#         contents_2 = f.read()
#         doc2.append(contents_2)

# Encapsulating file-reading into a function
def read_files(path, doc):
    for file_obj in glob.iglob(path):
        with open(file_obj, errors="ignore") as file_reader:
            contents = file_reader.read()
            doc.append(contents)

read_files(path_CVE_fixes_linux, doc1)
read_files(path_CVE_flaws_linux, doc2)

doc3 = pd.DataFrame(doc1)
doc4 = pd.DataFrame(doc2)

doc5 = doc3.to_string()
doc6 = doc4.to_string()

#-----------------------------------------------------------------------------------------#


def apply_preprocessing(data_instance):
    word_tokens = []
    for word in word_tokenize(str(data_instance)):
       if word not in stop_words_list:
           word_tokens.append(word)

    preprocessDataSet = json.dumps(word_tokens)
    preprocessDataSet = [preprocessDataSet]
    return preprocessDataSet

pp_doc1 = apply_preprocessing(doc5)
pp_doc2 = apply_preprocessing(doc6)



tf_idf_vectorizer = TfidfVectorizer()
feature_vectors = tf_idf_vectorizer.fit_transform([str(pp_doc1), str(pp_doc2)])
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

print(flaw_df.nlargest(20))


scaler = StandardScaler() # test
scaler.fit(X) # test
X = scaler.transform(X) # test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
clf_GNB = GaussianNB()
clf_RF = RandomForestClassifier(n_estimators=100)
clf_LR = LogisticRegression(max_iter=1000)
clf_KNN = KNeighborsClassifier(n_neighbors=3)
clf_MLP = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)

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
print(flaw_df.nlargest(20))

# Dump into Pickle file for flask deployment
clf_GNB.fit(X, y)
pickle.dump(clf_GNB, open('KFold_GNB_Linux.pkl','wb'))


