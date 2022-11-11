#!/usr/bin/env python
# coding: utf-8

# In[24]:


#!pip install streamlit


# In[26]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st


# In[22]:


unemp = pd.read_csv('unemployment analysis.csv')


# In[6]:


unemp_long= pd.melt(unemp, id_vars=['Country Name','Country Code'], var_name="Year", value_name= 'Unemployment Rate')
unemp_long


# In[7]:


unemp.drop(columns = ['Country Code'], inplace = True)
unemp.set_index('Country Name', inplace = True)


# In[8]:


df1 = unemp.T
df1


# In[9]:


fig = px.line(df1[['Africa Eastern and Southern', 'Africa Western and Central',"Middle East & North Africa", 'Central Europe and the Baltics',
                    "Europe & Central Asia",
                    'East Asia & Pacific', "Latin America & Caribbean", 'United States', 'Australia']])
fig.show()


# In[10]:


highun=df1.mean(axis=0).nlargest(10)
highun


# In[11]:


df3 = {'Lesotho':30.396452 , 'North Macedonia':29.789677 , 'South Africa':28.232581 , 'Djibouti':27.733226, 
       'Eswatini':24.391290, 'Bosnia and Herzegocina':24.044516, 'Montenegro':23.048387, 'Namibia':21.033548, 
       'Cong, Rep.':20.291613, 'Botswana':19.814839}


# In[12]:


courses = list(df3.keys())
values = list(df3.values())
  
fig = plt.figure(figsize = (10, 5))
 
# creating the bar plot
plt.bar(courses, values, color ='r',
        width = 0.4)
 
plt.xticks(rotation = 45) 
plt.ylabel("% Unemployed")
plt.title("Highest Unemployed Countries")
plt.show()


# In[13]:


lowun=df1.mean(axis=0).nsmallest(10)
lowun


# In[14]:


df5 = {'Qatar':0.569355, 'Cambodia':0.767419, 'Myanmar':0.916774, 'Rwanda':0.916774, 
       'Chad':0.950000, 'Bahrain':1.164839, 'Thailand':1.315806, 'benin':1.346452, 
       'Solomon Islands':1.370645, 'Niger':1.376452}


# In[15]:


courses = list(df5.keys())
values = list(df5.values())
  
fig = plt.figure(figsize = (10, 5))
 
# creating the bar plot
plt.bar(courses, values, color ='r',
        width = 0.4)
 
plt.xticks(rotation = 45) 
plt.ylabel("% Unemployed")
plt.title("Highest Unemployed Countries")
plt.show()


# In[16]:


gk = unemp_long.groupby('Year')
gk.get_group('1991')


# In[17]:


sns.scatterplot(x="Year", y="Unemployment Rate", data=unemp_long)
plt.title("Unemployment Rate", size=20, color="red")
plt.xlabel("Year")
plt.xticks(rotation = 90) 
#plt.ylabel("Unemployment Rate")
plt.show()


# In[27]:


st.set_page_config(page_title="Dashboard Groep 22", page_icon="🚗", layout = "wide", initial_sidebar_state="expanded")


# In[28]:


st.title('Dashboard Elektrische Mobiliteit en Laadpalen')


# In[29]:


st.sidebar.title('Navigatie')


# In[ ]:




