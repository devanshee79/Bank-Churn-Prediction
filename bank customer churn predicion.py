#!/usr/bin/env python
# coding: utf-8

# In[137]:


import numpy as np
import pandas as pd
import seaborn as sns

data = pd.read_csv(r'F:\Analyst projects\churn prediction\Churn_Modelling.csv')


# In[138]:


data.head()


# In[139]:


data.shape


# In[140]:


print('No.of rows: ', data.shape[0])
print('No.of cols: ', data.shape[1])


# In[141]:


data.info()


# In[142]:


data.describe()


# In[143]:


data.isnull().sum()


# In[144]:


# As rownum, customer_id and surname will not be having any impact on prediction; dropping them
data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis = 1)


# In[145]:


data['Geography'].unique()


# In[146]:


data = pd.get_dummies(data, drop_first=True)
data.head()


# In[147]:


data['Exited'].value_counts()


# In[148]:


sns.countplot(data['Exited'])


# In[149]:


X = data.drop('Exited', axis = 1)
y = data['Exited']
y


# In[150]:


print(X.shape)
print(len(y))


# In[151]:


pip install imbalanced-learn


# In[158]:


# As imbalenced data is there we have to balance it
from imblearn.over_sampling import SMOTE

X_res, y_res = SMOTE().fit_resample(X,y)


# In[159]:


X_res.value_counts()


# In[160]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)


# In[161]:


# Feature scaling
from sklearn.preprocessing import StandardScaler 
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[ ]:





# In[ ]:





# # Logistic regression

# In[162]:


from sklearn.linear_model import LogisticRegression
log = LogisticRegression()
log.fit(X_train,y_train)
y_pred1 = log.predict(X_test)


# In[163]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred1)


# In[164]:


from sklearn.metrics import precision_score, recall_score, f1_score
precision_score(y_test, y_pred1)


# In[165]:


recall_score(y_test, y_pred1)


# In[166]:


f1_score(y_test, y_pred1)


# # SVC

# In[167]:


from sklearn import svm
svm = svm.SVC ()
svm.fit(X_train, y_train)
y_pred2 = svm.predict (X_test)


# In[168]:


accuracy_score (y_test, y_pred2)


# In[169]:


precision_score(y_test, y_pred2)


# # KNeighbour Classifier

# In[176]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier ()
knn.fit (X_train, y_train)


# In[179]:


y_pred3 = knn.predict (X_test)


# In[180]:


precision_score(y_test, y_pred3)


# In[181]:


accuracy_score (y_test, y_pred3)


# # Decision Tree Classifier

# In[183]:


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier ()
dt.fit (X_train, y_train)


# In[185]:


y_pred4= dt.predict(X_test)


# In[186]:


accuracy_score (y_test, y_pred4)


# In[187]:


precision_score(y_test, y_pred4)


# # RandomForestClassifier

# In[188]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit (X_train, y_train)


# In[190]:


y_pred5 = rf.predict (X_test)
accuracy_score (y_test, y_pred5)


# In[191]:


precision_score(y_test, y_pred5)


# # GradientBoostingClassifier

# In[194]:


from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()
gbc.fit(X_train, y_train)


# In[195]:


y_pred6 = rf.predict(X_test)
accuracy_score (y_test, y_pred6)


# In[196]:


precision_score(y_test, y_pred5)


# In[207]:


final_data=pd.DataFrame({ 'Models': ['LR', 'SVC', 'KNN', 'DT', 'RF', 'GBC'], 'ACC': [accuracy_score (y_test, y_pred1),
                                                                                    accuracy_score (y_test, y_pred2),
                                                                                     accuracy_score (y_test, y_pred3),
                                                                                     accuracy_score (y_test, y_pred4),
                                                                                     accuracy_score (y_test, y_pred5),
                                                                                     accuracy_score (y_test, y_pred6)]})


# In[208]:


sns.barplot (final_data['Models'], final_data['ACC'])


# In[201]:


final_data=pd.DataFrame({ 'Models': ['LR', 'SVC', 'KNN', 'DT', 'RF', 'GBC'], 'PRE': [precision_score (y_test, y_pred1),
                                                                                    precision_score (y_test, y_pred2),
                                                                                     precision_score (y_test, y_pred3),
                                                                                     precision_score (y_test, y_pred4),
                                                                                     precision_score (y_test, y_pred5),
                                                                                     precision_score (y_test, y_pred6)]})


# In[204]:


sns.barplot (final_data['Models'], final_data['PRE'])


# In[198]:


X_res = sc.fit_transform(X_res)
rf.fit(X_res, y_res)


# In[199]:


import joblib


# In[200]:


joblib.dump(rf,'churn_predict_model')


# In[209]:


model = joblib.load('churn_predict_model')


# In[210]:


data.columns


# In[214]:


prediction = model.predict([[619, 42,2, 0.0,0,0,0,101348.88,0,0,0]])


# In[215]:


print(prediction)


# In[216]:


probabilities = model.predict_proba([[619, 42, 2, 0.0, 0, 0, 0, 101348.88, 0, 0, 0]])
print(probabilities)

