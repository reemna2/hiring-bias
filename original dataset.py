#!/usr/bin/env python
# coding: utf-8

# In[93]:


import numpy as np # linear algebra
import pandas as pd # data processing, 

# Libraries for data visualization
import matplotlib.pyplot as pplt  
import seaborn as sns 
from pandas.plotting import scatter_matrix

# Import scikit_learn module for the algorithm/model: Linear Regression
from sklearn.linear_model import LogisticRegression
# Import scikit_learn module to split the dataset into train.test sub-datasets
from sklearn.model_selection import train_test_split 
# Import scikit_learn module for k-fold cross validation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
# import the metrics class
from sklearn import metrics
# import stats for accuracy 
import statsmodels.api as sm
import numpy
import pandas
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier


import matplotlib.pyplot as plt
#from pandas.tools.plotting import scatter_matrix

from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error



from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Activation
from keras.constraints import maxnorm


# In[94]:


df = pd.read_csv(r'C:\Users\reemn\Downloads\adult.csv\adult.csv')


# In[95]:


df.rename(columns={"capital.gain": "capital gain", 'capital.loss': 'capital loss', 'native.country': 'country','hours.per.week': 'hours per week','marital.status': 'marital'}, inplace=True)
df.columns


# In[96]:


#Finding the special characters in the data frame 
df.isin(['?']).sum(axis=0)


# In[97]:


# code will replace the special character to nan and then drop the columns 

df['workclass'] = df['workclass'].replace('?',np.nan)
df['occupation'] = df['occupation'].replace('?',np.nan)
df['country'] = df['country'].replace('?',np.nan)
#dropping the NaN rows now 
df.dropna(how='any',inplace=True)


# In[98]:


#dropping based on uniquness of data from the dataset 
df.drop(['education.num','age', 'hours per week', 'fnlwgt', 'capital gain','capital loss', 'country'], axis=1, inplace=True)


# In[99]:


df = df.dropna()
df['income'] = df['income'].map({'<=50K': 0, '>50K': 1}).astype(int)


# In[100]:


#gender
df['sex'] = df['sex'].map({'Male': 0, 'Female': 1}).astype(int)
#race
df['race'] = df['race'].map({'Black': 0, 'Asian-Pac-Islander': 1,'Other': 2, 'White': 3, 'Amer-Indian-Eskimo': 4}).astype(int)
#marital
df['marital'] = df['marital'].map({'Married-spouse-absent': 0, 'Widowed': 1, 'Married-civ-spouse': 2, 'Separated': 3, 'Divorced': 4,'Never-married': 5, 'Married-AF-spouse': 6}).astype(int)


# In[101]:


#workclass
df['workclass'] = df['workclass'].map({'Self-emp-inc': 0, 'State-gov': 1,'Federal-gov': 2, 'Without-pay': 3, 'Local-gov': 4,'Private': 5, 'Self-emp-not-inc': 6}).astype(int)


# In[102]:


#education
df['education'] = df['education'].map({'Some-college': 0, 'Preschool': 1, '5th-6th': 2, 'HS-grad': 3, 'Masters': 4, '12th': 5, '7th-8th': 6, 'Prof-school': 7,'1st-4th': 8, 'Assoc-acdm': 9, 'Doctorate': 10, '11th': 11,'Bachelors': 12, '10th': 13,'Assoc-voc': 14,'9th': 15}).astype(int)


# In[103]:


#occupation
df['occupation'] = df['occupation'].map({ 'Farming-fishing': 1, 'Tech-support': 2, 'Adm-clerical': 3, 'Handlers-cleaners': 4, 
 'Prof-specialty': 5,'Machine-op-inspct': 6, 'Exec-managerial': 7,'Priv-house-serv': 8,'Craft-repair': 9,'Sales': 10, 'Transport-moving': 11, 'Armed-Forces': 12, 'Other-service': 13,'Protective-serv':14}).astype(int)


# In[104]:


#relationship
df['relationship'] = df['relationship'].map({'Not-in-family': 0, 'Wife': 1, 'Other-relative': 2, 'Unmarried': 3,'Husband': 4,'Own-child': 5}).astype(int)


# In[109]:


plt.figure(figsize=(12,6));
sns.histplot(binwidth=0.5, x="sex", hue="income", data=df, stat="count", multiple="stack");


# In[110]:


df_men, df_women = [x for _, x in df.groupby(df['sex'] == 1 )]


# In[111]:


df_men.info()


# In[ ]:





# In[112]:


df_men.head()


# In[113]:


df_women.info()


# In[114]:


df_x_men = pd.DataFrame(np.c_[df_men['relationship'], df_men['education'], df_men['race'],df_men['occupation'],df_men['sex'],df_men['marital'],df_men['workclass']], columns = ['relationship','education','race','occupation','sex','marital','workclass'])
#Y axis = Our dependent variable or the income of adult i.e Income
df_y_men = pd.DataFrame(df_men.income)


# In[115]:


df_x_women = pd.DataFrame(np.c_[df_women['relationship'], df_women['education'], df_women['race'],df_women['occupation'],df_women['sex'],df_women['marital'],df_women['workclass']], columns = ['relationship','education','race','occupation','sex','marital','workclass'])
#Y axis = Our dependent variable or the income of adult i.e Income
df_y_women = pd.DataFrame(df_women.income)


# In[116]:


#Initialize the linear regression model
reg = LogisticRegression()
#Split the data into 77% training and 33% testing data
#NOTE: We have to split the dependent variables (x) and the target or independent variable (y)
x_train_men, x_test_men, y_train_men, y_test_men = train_test_split(df_x_men, df_y_men, test_size=0.33, random_state=42)


# In[117]:


#Initialize the linear regression model
#reg = LogisticRegression()
#Split the data into 77% training and 33% testing data
#NOTE: We have to split the dependent variables (x) and the target or independent variable (y)
x_train_women, x_test_women, y_train_women, y_test_women = train_test_split(df_x_women, df_y_women, test_size=0.33, random_state=42)


# In[118]:


#num_instances = len(X)

models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('L_SVM', LinearSVC()))
models.append(('SGDC', SGDClassifier()))

# Evaluations
results = []
names = []

for name, model in models:
    # Fit the model
    model.fit(x_train_men, y_train_men)
    
    predictions = model.predict(x_test_men)
    
    # Evaluate the model
    score = accuracy_score(y_test_men, predictions)
    mse = mean_squared_error(predictions, y_test_men)
    # print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    results.append(mse)
    names.append(name)
    
    msg = "%s: %f (%f)" % (name, score, mse)
    print(msg)


# In[119]:


models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('L_SVM', LinearSVC()))
models.append(('SGDC', SGDClassifier()))

# Evaluations
results = []
names = []

for name, model in models:
    # Fit the model
    model.fit(x_train_women, y_train_women)
    
    predictions = model.predict(x_test_women)
    
    # Evaluate the model
    score = accuracy_score(y_test_women, predictions)
    mse = mean_squared_error(predictions, y_test_women)
    # print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    results.append(mse)
    names.append(name)
    
    msg = "%s: %f (%f)" % (name, score, mse)
    print(msg)


# In[ ]:


num_instances = len(df_x_women)

models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('L_SVM', LinearSVC()))
models.append(('SGDC', SGDClassifier()))

# Evaluations
results = []
names = []

for name, model in models:
    # Fit the model
    model.fit(x_train_women, y_train_women)
    
    predictions = model.predict(x_test_women)
    
    # Evaluate the model
    score = accuracy_score(y_test_women, predictions)
    mse = mean_squared_error(predictions, y_test_women)
    # print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    results.append(mse)
    names.append(name)
    
    msg = "%s: %f (%f)" % (name, score, mse)
    print(msg)


# In[120]:


#Train our model with the training data
reg.fit(x_train_men, y_train_men)
#print our price predictions on our test data
y_pred_men = reg.predict(x_test_men)


# In[121]:


#Train our model with the training data
reg.fit(x_train_women, y_train_women)
#print our price predictions on our test data
y_pred_women = reg.predict(x_test_women)


# In[122]:


#feeding the predict function with our test values in the format 
[['relationship','education','race','occupation','gender','marital','workclass']]
reg.predict([[1,7,3,7,0,2,0]])


# In[123]:


#feeding the predict function with our test values in the format 
[['relationship','education','race','occupation','gender','marital','workclass']]
reg.predict([[1,7,3,7,0,2,0]])


# In[124]:


#printing the accuracy values 
print("Accuracy:",metrics.accuracy_score(y_test_men, y_pred_men))


# In[125]:


#printing the accuracy values 
print("Accuracy:",metrics.accuracy_score(y_test_women, y_pred_women))


# In[126]:


from sklearn.metrics import classification_report


# In[127]:


print(classification_report(y_test_men, y_pred_men))


# In[128]:


print(classification_report(y_test_women, y_pred_women))

