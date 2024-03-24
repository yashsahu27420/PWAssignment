#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""Q1. What are the key features of the wine quality data set? Discuss the importance of each feature in
predicting the quality of wine."""


# In[1]:


"""the key feature in thye wine quality dataset is the there categorical featuire  and their numerical feature where it contain fixid oil , volatile oil,citric acid,resudual sugar,cl,H2so4,so2,ph,s3,oH,etc"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt ## importing imp libaray for furthere refernece
import seaborn as sns
df = pd.read_csv('WineQT.csv')
df.head(2)
df.drop('Id',axis=1,inplace=True)
df.head(2)


# In[2]:


df.info() #""" cloumns having the key features for analysing the wine quality in this data set and coralating these factor help to visulazing the compents play major role inthebquality of wine"""


# In[3]:


plt.figure(figsize=(10,8))
sns.heatmap(data=df,annot=True,cmap='coolwarm',cbar_kws={'format': '%.2f'},annot_kws={'size': 12})
plt.show()


# In[4]:


"""Q2. How did you handle missing data in the wine quality data set during the feature engineering process?
Discuss the advantages and disadvantages of different imputation techniques."""


# In[4]:


"""we'll handle missing data by using isnull and duplicated function """
df.isnull().sum() ## here there is no missing data or null value found


# In[5]:


df.duplicated().shape ## but this much of duplicated variable i found 


# In[6]:


df.drop_duplicates()


# In[7]:


df.head()


# In[15]:


"""3. Mode Imputation (for Categorical Variables):
Advantages:

Simple and appropriate for categorical variables.
Preserves the mode of the data.
Disadvantages:

May not be suitable for variables with multiple modes or if the mode is not representative of the missing value."""


# In[16]:


"""Q3. What are the key factors that affect students' performance in exams? How would you go about
analyzing these factors using statistical techniques?"""


# In[8]:


df1 = pd.read_csv('Student Performance.csv')
df1.head(2)


# In[9]:


df1.describe() # key factor are math score reading score,writting score affect the performance


# In[10]:


df1.info()


# In[25]:


"""Q4. Describe the process of feature engineering in the context of the student performance data set. How
did you select and transform the variables for your model?"""


# In[11]:


df1.head(2)


# In[12]:


#agregate the total score with mean:
df1['total_score'] = (df1['math score']+df1['reading score']+df1['writing score'])
df1['avearge'] = df1['total_score']/3
df1.head()
   


# In[13]:


df1['math score'].value_counts().sum()


# In[14]:


df1['reading score'].value_counts().sum()


# In[36]:


"""transforming dat into two categories for feature selection
1.numerical_features
2.categorical features"""


# In[15]:


numerical_features = [features for features in df1.columns if df1[features].dtype!='O']
categorical_features = [features for features in df1.columns if df1[features].dtype=='O']


# In[16]:


numerical_features


# In[17]:


categorical_features ## for this feature we can do one hot  encoding


# In[42]:


"""Q5. Load the wine quality data set and perform exploratory data analysis (EDA) to identify the distribution
of each feature. Which feature(s) exhibit non-normality, and what transformations could be applied to
these features to improve normality?"""


# In[18]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt ## importing imp libaray for furthere refernece
import seaborn as sns
df = pd.read_csv('WineQT.csv')
df.head(2)
df.drop('Id',axis=1,inplace=True)
df.head(2)


# In[19]:


df.info()


# In[20]:


plt.figure(figsize=(10,6))
sns.barplot(data=df,x='quality',y='residual sugar',color='g')
plt.show()


# In[21]:


plt.figure(figsize=(10,6))
sns.barplot(data=df,x='quality',y='fixed acidity',color='r')
plt.show()


# In[22]:


plt.figure(figsize=(10,6))
sns.histplot(data=df,x='quality',y='citric acid',color='r')
plt.show()


# In[23]:


import matplotlib.pyplot as plt

# Assuming 'df' is your DataFrame
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(16, 12))

for i, col in enumerate(df.columns[:-1]):  # Exclude the 'quality' column
    ax = axes[i//4, i%4]
    ax.scatter(df[col], df['quality'])
    ax.set_xlabel(col)
    ax.set_ylabel('Quality')

plt.tight_layout()
plt.show()


# EDA exploration

# In[26]:


"""fixcid oil : when it is 12 -14 the quality is max that time
volatile acidity : 0.25 -1.0 give the max quality 
citric acid: 0.2-0.8 citric acid give good quality
residual sugar: from 2.5-7.5 it give good quality to the wine
chloride:0 -0.1 is optimum range fro the quality
free sulfur dioxide:1-20 is optimum range for the quality
density:0.9925-1 give good quality to the wine
ph:ranges from 2.9-3.7 is better ph for the quality
sulphates:0.6-1.1 is good for the quality
alcohol:10-14 is better range for alchohal"""


# In[27]:


"""Q6. Using the wine quality data set, perform principal component analysis (PCA) to reduce the number of
features. What is the minimum number of principal components required to explain 90% of the variance in
the data?"""


# In[24]:


df.head(2)


# In[25]:


df.columns


# In[26]:


df.info()


# In[27]:


## standardization of data
mean =np.mean(df,axis=0)
std_dev=np.std(df,axis=0)
standardize_data=(df-mean)/std_dev


# In[28]:


standardize_data


# In[29]:


## claculating covariance
covariance_matrix=np.cov(df,rowvar=False)


# In[30]:


covariance_matrix


# In[31]:


eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)


# In[32]:


eigenvalues


# In[33]:


eigenvectors


# In[35]:


sorted_eigenvalues = np.sort(eigenvalues)[::-1]


# In[36]:


sorted_eigenvalues


# In[37]:


total_variance = np.sum(sorted_eigenvalues)


# In[38]:


variance_proportion = sorted_eigenvalues/total_variance


# In[39]:


cumulative_variance = np.cumsum(variance_proportion)


# In[40]:


cumulative_variance


# In[41]:


commponent_90 = np.argmax(cumulative_variance >= 0.9) + 1


# In[42]:


commponent_90


# In[49]:


selected_eigenvectors = sorted_eigenvalues[:commponent_90]


# In[ ]:




