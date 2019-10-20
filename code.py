
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from pylab import rcParams
get_ipython().run_line_magic('matplotlib', 'inline')
# Loading the CSV with pandas
data = pd.read_csv('\\projects\\Churn.csv')
data.head()


# In[2]:


# Data to plot
sizes = data['Churn'].value_counts(sort = True)
colors = ["yellow","red"] 
rcParams['figure.figsize'] = 5,5
explode = (0.1, 0)  # explode 1st slice
labels= 'no','yes'
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,autopct='%1.1f%%', shadow=True, startangle=270,)
plt.title('Percentage of Churn in Dataset')
plt.show()


# In[3]:


data.drop(['customerID'], axis=1, inplace=True)
data['MonthlyCharges'] = pd.to_numeric(data['MonthlyCharges'])
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'])
data.head()


# In[6]:


## Examine correlations
data['tenuremonth'] = (data['tenure'] * data['MonthlyCharges']).astype(float)
data.corr()


# In[7]:


# 6 features, convert 'no internet service' to 'no'
no_int_service_vars = ['OnlineSecurity', 'OnlineBackup', 
                       'DeviceProtection','TechSupport', 
                       'StreamingTV', 'StreamingMovies']
                       
for var in no_int_service_vars:
    data[var] = data[var].map({'No internet service': 'No',
                           'Yes': 'Yes',
                           'No': 'No'}).astype('category')
    
for var in no_int_service_vars:
    print(data[var].value_counts())


# In[8]:


## Binarize binary variables
from sklearn.preprocessing import StandardScaler, LabelEncoder
df_enc = data.copy()
binary_vars = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
               'PhoneService', 'MultipleLines', 'OnlineSecurity', 
               'OnlineBackup','DeviceProtection', 'TechSupport', 
               'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'Churn']
enc = LabelEncoder()
df_enc[binary_vars] = df_enc[binary_vars].apply(enc.fit_transform)

## One-hot encode multi-category cat. variables
multicat_vars = ['InternetService', 'Contract', 'PaymentMethod']
df_enc = pd.get_dummies(df_enc, columns = multicat_vars)
df_enc.iloc[:,16:26] = df_enc.iloc[:,16:26].astype(int)
print(df_enc.info())


# In[9]:


Y = df_enc["Churn"].values
X = df_enc.drop(labels = ["Churn"],axis = 1)
# Create Train & Test Data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.43, random_state=101)


# In[10]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
result = model.fit(X_train, y_train)


# In[11]:


from sklearn import metrics
prediction_test = model.predict(X_test)
print (metrics.accuracy_score(y_test, prediction_test))


# In[12]:


# To get the weights of all the variables
weights = pd.Series(model.coef_[0],index=X.columns.values)
weights.sort_values(ascending = False)

