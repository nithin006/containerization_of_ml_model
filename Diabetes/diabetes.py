#!/usr/bin/env python
# coding: utf-8

# In[45]:


import numpy as np
import pandas as pd
import seaborn as sns
import pickle


# In[46]:


df = pd.read_csv("diabetes.csv")


# In[47]:


df


# In[48]:


df.isna().sum()


# In[49]:


df.dtypes


# In[50]:


X = df.drop("Outcome",axis = 1)


# In[51]:


y = df["Outcome"]


# In[52]:


y


# In[53]:


from sklearn.model_selection import train_test_split


# In[55]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,random_state =123)


# In[56]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train,y_train)


# In[57]:


print("Saving model as pkl files...")
pickle.dump(lr,open('model.pkl','wb'))


# In[58]:


model = pickle.load(open('model.pkl','rb'))


# In[ ]:




