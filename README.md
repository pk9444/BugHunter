# BugHunter
## Master's Capstone Senior Project - "Vulnerability Detection Using Machine Learning"

### Description
BugHunter is a web app that inputs C language code snippet, converts it into feature vectore and based on a trained Machine Learning pipeline, performs prediction on whether the feature vector (C-Keyword) is vulnerable or not. If a possible vulnerability, then it is predicted as '1' else predicted as '0'. 

The pipeline has been validated through K-Fold cross validation over five C-source code datasets that have vulnerabilities as per the CVE (Common Vulnerability Enumneration) standards and also their patched datasets as recommended fixes for a given CVE. The cross validation has been performed using 5 different ML-algorithms which are : 

- Logistic Regression 
- Gaussian Naive-Bayes 
- K-Nearest Neighbors 
- Random Forest 
- Multilayer Perceptron 

Any ML-pipline can be deployed to production through Flask and it can be tested on a given N-gram as defined in the web server code. 

### Technology Stack 

- Python (3.9 64-bit)
- HTML
- CSS
- Flask

### Read Me 

#### Required python libraries 
- scikit-learn
- NumPy
- SciPy
- Pandas
- NLTK
- Flask 

- Install all the libraries in your compiler or simply ```pip install``` in the command prompt of your project directory 
- atleast run once any one of the machine learning model .py so that the .pkl (pickle) files are generated 
- you may use the given .pkl files but can generate your own if you want - just follow the steps in the code 
- run app.py 
- click on the web address - input your C-code snippet and detect possible vulnerabilities in your C code as detected by the trained model as read from the given .pkl file

