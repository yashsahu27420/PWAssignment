#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[33]:


pd.DataFrame(encoder.fit_transform(df[['Airline','Source','Destination']]).toarray().astype(int),columns=encoder.get_feature_names_out())


# In[35]:


df.info()


# In[30]:


airline_feature_names = encoder.get_feature_names_out(['Airline'])


# In[31]:


airline_feature_names


# In[36]:


encoded_df = pd.DataFrame(encoded_Airline, columns=airline_feature_names)

# Concatenate the encoded DataFrame with the original DataFrame
df_encoded = pd.concat([df, encoded_df], axis=1)

# Drop the original 'Airline' column if needed
df_encoded.drop(['Airline'], axis=1, inplace=True)


# In[39]:


df.head(2)


# In[41]:


"""Q5. Are there any outliers in the dataset? Identify any potential outliers using a boxplot and describe how
they may impact your analysis."""


# In[43]:


df.isnull().sum() # we can use mode to handle outliers 


# In[44]:


df['Total_Stops'].unique()


# In[48]:


df.drop('Route',axis=1,inplace=True)


# In[51]:


"""here 80,000 is the outlier Impact on Analysis:

Outliers can potentially have a significant impact on your analysis. They might skew summary statistics like the mean and standard deviation, leading to misleading conclusions. It's important to consider whether these outliers represent genuine data points or if they are errors or anomalies that need to be addressed."""


# In[52]:


"""Q6. You are working for a travel agency, and your boss has asked you to analyze the Flight Price dataset
to identify the peak travel season. What features would you analyze to identify the peak season, and how
would you present your findings to your boss?"""


# In[4]:


##rest our index to seperate day,date &year

df['day'] = df['Date_of_Journey'].str.split('/').str[0]
df['month'] = df['Date_of_Journey'].str.split('/').str[1]
df['year'] = df['Date_of_Journey'].str.split('/').str[2]


# In[8]:


df['day'].astype(int)
df['month'].astype(int)
df['year'].astype(int)


# In[7]:


df.head(2)


# goggle playstore

# In[42]:


"""Q9. Load the Google Playstore dataset and examine its dimensions. How many rows and columns does
the dataset have?"""


# In[4]:


df1 = pd.read_csv('https://raw.githubusercontent.com/krishnaik06/playstore-Dataset/main/googleplaystore.csv')
df1.head(2)


# In[11]:


df1.shape ## 10841 rows and 13 columns


# In[12]:


df.info()


# In[13]:


df1.columns


# In[50]:


"""Q10. How does the rating of apps vary by category? Create a boxplot to compare the ratings of different
app categories."""


# In[14]:


df1.head(2)


# In[15]:


from sklearn.preprocessing import OneHotEncoder


# In[16]:


##creating instance
encoder = OneHotEncoder()
encoder.fit_transform(df1[['App','Category']]).toarray()


# In[17]:


pd.DataFrame(encoder.fit_transform(df1[['App','Category']]).toarray().astype(int),columns=encoder.get_feature_names_out())


# In[18]:


df1.head(2)


# In[19]:


df1['Rating'].unique().astype(float)


# In[20]:


df1.info()


# In[61]:


plt.figure(figsize=(10,8))
sns.boxplot(data=df1,x='Rating',y='Category',color='g')
plt.tittle('rating compared to category')
plt.xlabel('rating')
plt.ylabel('categories')
plt.show()


# In[62]:


"""Q11. Are there any missing values in the dataset? Identify any missing values and describe how they may
impact your analysis."""


# In[21]:


df1.isnull().sum() ## mising value


# In[65]:


""" they can alter the prediction of any statical analysis by their presence """


# In[66]:


"""Q12. What is the relationship between the size of an app and its rating? Create a scatter plot to visualize
the relationship."""


# In[22]:


df1.head(2)


# In[23]:


df1['Size'].unique()


# In[24]:


df1['Size'] = df1['Size'].str.replace('M','000')
df1['Size'] = df1['Size'].str.replace('k','')
df1['Size'] = df1['Size'].replace('Varies with device',np.nan)
df1['Size'] = df1['Size'].str.replace(',','')
df1['Size'] = df1['Size'].str.replace('+', '')


# In[25]:


df1['Size'].astype(float)


# In[7]:


plt.figure(figsize=(10,8))
sns.scatterplot(data=df1,x='Size',y='Category',color='r')
plt.title('size vs category')
plt.xlabel('size')
plt.ylabel('category')
plt.show()


# In[8]:


"""Q13. How does the type of app affect its price? Create a bar chart to compare average prices by app type."""


# In[26]:


df1.head(2)


# In[27]:


df1.info()


# In[28]:


df1['Price'].unique()


# In[29]:


chars_to_remove=['+','$',',']
cols_to_clean=['Price','Installs']
for item in chars_to_remove:
    for cols in cols_to_clean:
        df1[cols]=df1[cols].str.replace(item,'')


# In[30]:


df1['Price'] = df1['Price'].replace('Everyone',np.nan)


# In[31]:


df1['Installs']= df1['Installs'].replace('Free','0')


# In[32]:


df1['Installs'].unique()


# In[36]:


df1['Price'] = df1['Price'].astype(float)
df1['Installs'] = df1['Installs'].astype(int)


# In[37]:


df1.info()


# In[41]:


plt.figure(figsize=(10,8))
sns.barplot(data=df1,x='Price',y='Category',color='g')
plt.title('type of category vs price')
plt.xlabel('price')
plt.ylabel('category')
plt.show()


# In[42]:


"""observation: here fininace and life and fitness have the most high price in th etype of app"""


# In[43]:


"""Q14. What are the top 10 most popular apps in the dataset? Create a frequency table to identify the apps
with the highest number of installs."""


# In[44]:


## Top  App Categories
category = pd.DataFrame(df1['Category'].value_counts())        
category.rename(columns = {'Category':'Count'},inplace=True)


# In[45]:


category


# In[46]:


"""observation: family and game app have most no of installs"""


# In[47]:


"""Q15. A company wants to launch a new app on the Google Playstore and has asked you to analyze the
Google Playstore dataset to identify the most popular app categories. How would you approach this
task, and what features would you analyze to make recommendations to the company?"""


# In[48]:


df1['Category'].value_counts().plot.pie(y=df1['Category'],figsize=(15,16),autopct='%1.2f')


# In[49]:


"""Q16. A mobile app development company wants to analyze the Google Playstore dataset to identify the
most successful app developers. What features would you analyze to make recommendations to the
company, and what data visualizations would you use to present your findings?"""


# In[50]:


""" the most successful app developer is family app developer"""


# In[51]:


"""Q17. A marketing research firm wants to analyze the Google Playstore dataset to identify the best time to
launch a new app. What features would you analyze to make recommendations to the company, and
what data visualizations would you use to present your findings?"""


# In[53]:


df1.head(2)


# In[57]:


df1['Last Updated']=pd.to_datetime(df1['Last Updated'],errors='coerce')
df1['Day']=df1['Last Updated'].dt.day
df1['Month']=df1['Last Updated'].dt.month
df1['Year']=df1['Last Updated'].dt.year


# In[66]:


df1.head(2)


# In[59]:


df1.info()


# In[68]:


pivot_df = df1.pivot_table(index='Category', columns='Year', values='Price', aggfunc='mean')

plt.figure(figsize=(10,8))
sns.heatmap(pivot_df,cmap='Set1',annot=True, fmt=".1f", linewidths=.5, cbar_kws={'label': 'Price'}, 
            annot_kws={"size": 10}, square=True)
plt.title('price of category acc to year')
plt.xlabel('category')
plt.ylabel('year')
plt.show()


# In[ ]:




