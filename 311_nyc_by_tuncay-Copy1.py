#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><ul class="toc-item"><li><span><a href="#Question-1---Which-type-of-complaints-The-Department-of-Housing-Preservation-and-Development-of-New-York-City-should-focus-first?" data-toc-modified-id="Question-1---Which-type-of-complaints-The-Department-of-Housing-Preservation-and-Development-of-New-York-City-should-focus-first?-0.1">Question 1 - Which type of complaints The Department of Housing Preservation and Development of New York City should focus first?</a></span><ul class="toc-item"><li><span><a href="#Which-type-of-complaint-should-the-Department-of-Housing-Preservation-and-Development-of-New-York-City-focus-on-first?" data-toc-modified-id="Which-type-of-complaint-should-the-Department-of-Housing-Preservation-and-Development-of-New-York-City-focus-on-first?-0.1.1">Which type of complaint should the Department of Housing Preservation and Development of New York City focus on first?</a></span></li><li><span><a href="#What-is-the-total-number-of-General-Construction-complaints-that-came-to-the-Department-of-Housing-Preservation-and-Development-of-New-York-City?" data-toc-modified-id="What-is-the-total-number-of-General-Construction-complaints-that-came-to-the-Department-of-Housing-Preservation-and-Development-of-New-York-City?-0.1.2">What is the total number of General Construction complaints that came to the Department of Housing Preservation and Development of New York City?</a></span></li><li><span><a href="#For-the-Complaint-Type-that-you-selected,-can-you-determine-the-total-number-of-complaints-at-the-Street-level?" data-toc-modified-id="For-the-Complaint-Type-that-you-selected,-can-you-determine-the-total-number-of-complaints-at-the-Street-level?-0.1.3">For the Complaint Type that you selected, can you determine the total number of complaints at the Street level?</a></span></li><li><span><a href="#Which-approach-do-you-think-can-be-used-to-find-the-total-number-of-complaints-for-an-address?" data-toc-modified-id="Which-approach-do-you-think-can-be-used-to-find-the-total-number-of-complaints-for-an-address?-0.1.4">Which approach do you think can be used to find the total number of complaints for an address?</a></span></li><li><span><a href="#Which-is-the-Complaint-Type-that-the-Department-of-Housing-Preservation-and-Development-of-New-York-City-should-address-first-considering-complaints-created-till-31st-Dec-2018-?" data-toc-modified-id="Which-is-the-Complaint-Type-that-the-Department-of-Housing-Preservation-and-Development-of-New-York-City-should-address-first-considering-complaints-created-till-31st-Dec-2018-?-0.1.5">Which is the Complaint Type that the Department of Housing Preservation and Development of New York City should address first considering complaints created till 31st Dec 2018 ?</a></span></li></ul></li></ul></li><li><span><a href="#Question-2---What-Areas-Should-the-Agency-Focus-On?" data-toc-modified-id="Question-2---What-Areas-Should-the-Agency-Focus-On?-1">Question 2 - What Areas Should the Agency Focus On?</a></span><ul class="toc-item"><li><ul class="toc-item"><li><span><a href="#Should-the-Department-of-Housing-Preservation-and-Development-of-New-York-City-focus-on-any-particular-set-of-boroughs,-ZIP-codes,-or-street-(where-the-complaints-are-severe)-for-the-specific-type-of-complaints-you-identified-in-response-to-Question-1?" data-toc-modified-id="Should-the-Department-of-Housing-Preservation-and-Development-of-New-York-City-focus-on-any-particular-set-of-boroughs,-ZIP-codes,-or-street-(where-the-complaints-are-severe)-for-the-specific-type-of-complaints-you-identified-in-response-to-Question-1?-1.0.1">Should the Department of Housing Preservation and Development of New York City focus on any particular set of boroughs, ZIP codes, or street (where the complaints are severe) for the specific type of complaints you identified in response to Question 1?</a></span></li><li><span><a href="#Which-approach-do-you-think-can-be-used-to-find-the-total-number-of-complaints-for-an-address?" data-toc-modified-id="Which-approach-do-you-think-can-be-used-to-find-the-total-number-of-complaints-for-an-address?-1.0.2">Which approach do you think can be used to find the total number of complaints for an address?</a></span></li><li><span><a href="#where-homelessness-is-a-problem?" data-toc-modified-id="where-homelessness-is-a-problem?-1.0.3">where homelessness is a problem?</a></span></li></ul></li></ul></li><li><span><a href="#Question-3---What-Is-the-Relationship-between-Housing-Characteristics-and-Complaints?" data-toc-modified-id="Question-3---What-Is-the-Relationship-between-Housing-Characteristics-and-Complaints?-2">Question 3 - What Is the Relationship between Housing Characteristics and Complaints?</a></span><ul class="toc-item"><li><ul class="toc-item"><li><span><a href="#Does-the-Complaint-Type-that-you-identified-in-response-to-Question-1-have-an-obvious-relationship-with-any-particular-characteristic-or-characteristic-of-the-Houses?" data-toc-modified-id="Does-the-Complaint-Type-that-you-identified-in-response-to-Question-1-have-an-obvious-relationship-with-any-particular-characteristic-or-characteristic-of-the-Houses?-2.0.1">Does the Complaint Type that you identified in response to Question 1 have an obvious relationship with any particular characteristic or characteristic of the Houses?</a></span></li><li><span><a href="#Can-you-determine-the-age-of-the-building-from-the-PLUTO-dataset?" data-toc-modified-id="Can-you-determine-the-age-of-the-building-from-the-PLUTO-dataset?-2.0.2">Can you determine the age of the building from the PLUTO dataset?</a></span></li></ul></li></ul></li><li><span><a href="#Question-4---Predict-Complaint-Types" data-toc-modified-id="Question-4---Predict-Complaint-Types-3">Question 4 - Predict Complaint Types</a></span><ul class="toc-item"><li><ul class="toc-item"><li><span><a href="#Can-a-predictive-model-be-built-for-future-prediction-of-the-possibility-of-complaints-of-the-specific-type-that-you-identified-in-response-to-Question-1?" data-toc-modified-id="Can-a-predictive-model-be-built-for-future-prediction-of-the-possibility-of-complaints-of-the-specific-type-that-you-identified-in-response-to-Question-1?-3.0.1">Can a predictive model be built for future prediction of the possibility of complaints of the specific type that you identified in response to Question 1?</a></span><ul class="toc-item"><li><span><a href="#MultiLinear-Model......THIS-IS-DONE-JUST-TO-GIVE-AN-IDEA.-IT-WILL-BE-IMPROVED-LATER-ON." data-toc-modified-id="MultiLinear-Model......THIS-IS-DONE-JUST-TO-GIVE-AN-IDEA.-IT-WILL-BE-IMPROVED-LATER-ON.-3.0.1.1">MultiLinear Model......THIS IS DONE JUST TO GIVE AN IDEA. IT WILL BE IMPROVED LATER ON.</a></span></li><li><span><a href="#Decision-tree" data-toc-modified-id="Decision-tree-3.0.1.2">Decision tree</a></span></li><li><span><a href="#Random-Forest" data-toc-modified-id="Random-Forest-3.0.1.3">Random Forest</a></span></li></ul></li><li><span><a href="#Can-the-model-that-you-developed-use-Number-of-Floors-in-an-address-as-a-possible-predictive-feature?" data-toc-modified-id="Can-the-model-that-you-developed-use-Number-of-Floors-in-an-address-as-a-possible-predictive-feature?-3.0.2">Can the model that you developed use Number of Floors in an address as a possible predictive feature?</a></span></li></ul></li></ul></li></ul></div>

# In[422]:


import json
import requests
import numpy as np
import pandas as pd
from datetime import datetime
from datetime import date
import csv
import urllib.request as urllib2
import dask.dataframe as dd
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import metrics
import statsmodels.api as sm
import ijson
import time
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import re
import math
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices


# ### Question 1 - Which type of complaints The Department of Housing Preservation and Development of New York City should focus first?

# #### Which type of complaint should the Department of Housing Preservation and Development of New York City focus on first?
# 
# 

# In[2]:


file='C:/Users/tuncay/Documents/311_Service_Requests_from_2010_to_Present.csv'
df=pd.read_csv(file)


# In[40]:


df['Created Date'] = pd.to_datetime(df['Created Date'],errors="coerce")
df['Hour'] = df["Created Date"].dt.strftime('%H')    
df['Day'] = df["Created Date"].dt.strftime('%d')    
df['Month'] = df["Created Date"].dt.strftime('%m')    
df['Year'] = df["Created Date"].dt.strftime('%Y')    


# In[41]:


df.head(1)


# In[219]:


df.columns


# In[3]:


df.info()


# In[4]:


df.columns


# In[10]:


df.isnull().sum()


# In[13]:


#df2 = df.dropna(axis=0, how="any")
#df2.isnull().sum()
#list(df["Incident Zip"].unique())


# In[3]:


#df['Incident Zip'].str.findall('\w{5,}').str.join(' ')

#t = df["Complaint Type"].str.split(expand=True).stack()
#t.loc[t.str.len() <3 ].groupby(level=0).apply(' '.join)

#s=df["Incident Zip2"]
#df["Zip2"]=re.sub(r'[0-9]{,3}',r'',s)


# In[121]:


#df['Incident Zip2'] = pd.to_numeric(df['Incident Zip'], errors='coerce')
#df = df.dropna(subset=['Incident Zip2'])
#df['Incident Zip2'] = df['Incident Zip2'].astype(str)
#df["Incident Zip2"]=df['Incident Zip2'].str.extract('^(\d{5})$', expand=False)


# In[133]:


df['Incident Zip3'] = df['Incident Zip'].astype(str)
df["Incident Zip3"]=df['Incident Zip3'].str.extract(r'\b(\d{5})\b', expand=False)
df = df.dropna(subset=['Incident Zip3'])
#list(df["Incident Zip3"].unique())


# In[ ]:





# In[44]:


df["Complaint Type"]=df["Complaint Type"].astype("category")
df["Complaint Type 2"]=df["Complaint Type"].cat.codes

df["Agency"]=df["Agency"].astype("category")
df["Agency2"]=df["Agency"].cat.codes

df["Agency Name"]=df["Agency Name"].astype("category")
df["Agency Name2"]=df["Agency Name"].cat.codes

df["Borough"]=df["Borough"].astype("category")
df["Borough2"]=df["Borough"].cat.codes

df["Open Data Channel Type"]=df["Open Data Channel Type"].astype("category")
df["Open Data Channel Type2"]=df["Open Data Channel Type"].cat.codes

df["Park Facility Name"]=df["Park Facility Name"].astype("category")
df["Park Facility Name2"]=df["Park Facility Name"].cat.codes

df["Park Borough"]=df["Park Borough"].astype("category")
df["Park Borough2"]=df["Park Borough"].cat.codes


# In[141]:


df2=df[["Agency2","Agency Name2", "Complaint Type 2","Borough2","Open Data Channel Type2","Park Facility Name2",
        "Park Borough2",
        "Hour","Day","Month","Year","Incident Zip3"]]


# In[143]:


df2.isnull().sum()


# In[396]:


plt.rcParams["figure.figsize"] = (13, 7)
ax=sns.countplot(x='Complaint Type',data=df)
ax.set_xticklabels(ax.get_xticklabels(),fontsize=5, rotation=40, ha="right")
plt.tight_layout()
plt.show()

#g = sns.factorplot("Complaint Type", data=df, aspect=1.5, kind="count", color="b")
#g.set_xticklabels(rotation=30)


# In[5]:


df.groupby(["Borough"])["Complaint Type"].value_counts()


# In[82]:


pd.set_option('display.max_rows', 500)
df["Complaint Type"].value_counts()
# df["Complaint Type"].unique()


# In[402]:


#plt.rcParams["figure.figsize"] = (10, 20)
df['Complaint Type'].value_counts().plot(kind='barh',alpha=0.3, figsize=(9,50)) 
plt.show()


# In[9]:


df.groupby(["Complaint Type"])["City"].value_counts()


# In[ ]:





# #### What is the total number of General Construction complaints that came to the Department of Housing Preservation and Development of New York City?

# In[290]:


# df[df["Complaint Type"]=="GENERAL CONSTRUCTION"].count()
df[df["Complaint Type"]=="GENERAL CONSTRUCTION"]["Unique Key"].count()


# In[275]:


df.groupby(["Complaint Type"]).size().reset_index(name='counts')


# In[330]:


#df[df['Incident Zip'].notnull() & (df['Complaint Type']=='GENERAL CONSTRUCTION')].head()


# In[334]:


g = sns.FacetGrid(data=df2,col='Borough2')
g.map(plt.hist,'Complaint_Type_2')


# In[6]:


complaints=df.groupby(["Complaint Type","Incident Zip"])["Borough"].describe()
complaints.columns
complaints.sort_values(by="count",ascending=False)


# #### For the Complaint Type that you selected, can you determine the total number of complaints at the Street level?
# 

# In[97]:


df.groupby(["Complaint Type","Street Name"])["Unique Key"].count()


# In[276]:


df.groupby(["Complaint Type","Street Name"])["Unique Key"].sum()


# In[295]:


#SEDGWICK AVENUE
df[df["Street Name"]=="SEDGWICK AVENUE"]["Complaint Type"].value_counts()


# In[297]:


#SEDGWICK AVENUE
df[df["Street Name"]=="SEDGWICK AVENUE"]["Unique Key"].count()


# #### Which approach do you think can be used to find the total number of complaints for an address?
# 

# In[299]:


df.groupby(["Complaint Type","Incident Zip","Incident Address"])["Unique Key"].count()


# In[298]:


df[df["Incident Address"]=="1230 BROADWAY"]["Unique Key"].count()


# In[300]:


df[df["Incident Address"]=="1230 BROADWAY"]["Complaint Type"].value_counts()


# #### Which is the Complaint Type that the Department of Housing Preservation and Development of New York City should address first considering complaints created till 31st Dec 2018 ?

# In[254]:


#df["Year"]=df["Year"].astype(int)
#(df["Year"].loc["2010":"2018"]).value_counts()
by2019=df[df["Year"]==2019]
pd.set_option('display.max_rows', 500)
by2019.groupby(by2019["Complaint Type"])["Unique Key"].count()

# by 2019 grouped_by only


# In[220]:


df.groupby(df["Year"])["Complaint Type"].value_counts()


# In[ ]:





# ## Question 2 - What Areas Should the Agency Focus On?

# #### Should the Department of Housing Preservation and Development of New York City focus on any particular set of boroughs, ZIP codes, or street (where the complaints are severe) for the specific type of complaints you identified in response to Question 1?

# In[256]:


df["Complaint Type"].unique()


# In[258]:


q2=pd.DataFrame(df.groupby(["Complaint Type","Borough","Incident Zip", "Street Name"])["Descriptor"].count())
# q2=pd.DataFrame(df.groupby(["complaint_type","borough", "street_name"])["incident_zip"].count())
(q2.sort_values(by=['Descriptor'],ascending=False)).head()


# In[324]:


df2.columns


# In[327]:


sns.boxplot(x='Month',y='Complaint_Type_2',data=df2,palette='rainbow')


# In[362]:


# sns.stripplot(x="Month", y="Complaint_Type_2", data=df2,jitter=True,hue='Borough2',palette='Set1',split=True)


# In[ ]:





# #### Which approach do you think can be used to find the total number of complaints for an address?

# In[303]:


df[df["Incident Address"]=="230 PARK AVENUE"]["Complaint Type"].count()


# In[302]:


df[df["Incident Address"]=="230 PARK AVENUE"]["Complaint Type"].value_counts()


# In[64]:


pd.DataFrame(df.groupby(["Complaint Type","Incident Address","City","Location Type"])["Descriptor"].count())


# #### where homelessness is a problem?

# In[305]:


# Homeless Person Assistance 
df[df["Complaint Type"]=="Homeless Person Assistance"]["Incident Address"].value_counts().head()


# In[ ]:





# ## Question 3 - What Is the Relationship between Housing Characteristics and Complaints?
# 

# #### Does the Complaint Type that you identified in response to Question 1 have an obvious relationship with any particular characteristic or characteristic of the Houses?

# In[70]:


path=r'C:\Users\tuncay\OneDrive\Coding\IBM\capstone'
allFiles = glob.glob(path + "/*.csv")
frame = pd.DataFrame()
list_ = []
for file_ in allFiles:
    br = pd.read_csv(file_,index_col=None, header=0)
    list_.append(br)
boroughs = pd.concat(list_)


# In[342]:


# boroughs.head()


# In[263]:


#boroughs.info()


# In[262]:


#boroughs.isnull().sum()


# In[171]:


boroughs2=boroughs.dropna(axis=1)


# In[261]:


#boroughs2.isnull().sum()


# In[260]:


#boroughs2.info()


# In[175]:


drop_columns=["Borough","Version"]
boroughs2.drop(drop_columns, axis=1, inplace=True)


# In[181]:


boroughs3=boroughs.fillna(0)


# In[191]:


#boroughs3.isnull().sum(),boroughs3.info()
boroughs4=boroughs3._get_numeric_data()


# In[259]:


#boroughs2["Borough"]=df["Borough"].astype("category")
#boroughs2["Borough2"]=df["Borough"].cat.codes

#boroughs4.info()


# In[343]:


# df["Borough"].unique(),boroughs["Borough"].unique(),boroughs3["Borough"].unique(),boroughs4["BoroCode"].unique()


# In[344]:


# boroughs4.head()


# In[345]:


# df2.columns,boroughs4.columns


# In[209]:


boroughs4['ZipCode'] = boroughs4['ZipCode'].astype(str)
boroughs4["ZipCode"]=boroughs4['ZipCode'].str.extract(r'\b(\d{5})\b', expand=False)
boroughs4 = boroughs4.dropna(subset=['ZipCode'])
#boroughs4['ZipCode'] = boroughs4['ZipCode'].astype(int)
#list(boroughs4["ZipCode"].unique())


# In[211]:


# merged_df2_borough4=pd.merge(df2, boroughs4, left_on="Incident_Zip3", right_on="ZipCode")

# MEMORY ERROR !!!!!


# In[186]:


boroughs.groupby(["BldgClass","ZipCode","PolicePrct"])["LotArea"].mean().sort_values(ascending=False).head(10)


# In[187]:


boroughs3.groupby(["BldgClass","ZipCode","PolicePrct"])["BldgArea"].mean().sort_values(ascending=False).head(10)


# In[357]:


#boroughs[["BldgClass","ZipCode","PolicePrct"]].head()
# boroughs[boroughs["Borough"]=="QN"]


# In[358]:


sns.jointplot(x='BldgArea',y='PolicePrct',data=boroughs)


# In[355]:


g = sns.FacetGrid(data=boroughs,col='Borough')
g.map(plt.hist,'BldgArea')


# In[70]:


boroughs.groupby(["BldgClass","ZipCode"])["Address"].count().sort_values(ascending=False)

# A is for one family dwellings
# B is for two family dwellings

# certain zipcodes have higher A and B building_class, can be used as a comparison, 
# such as what category of building_class has a higher correlation with what complaint_type.

# BUT DUE TO MEMORY ERROR, UNABLE TO DO SO.


# In[ ]:





# #### Can you determine the age of the building from the PLUTO dataset?
# 

# In[312]:


now=datetime.now()
boroughs["ageofbldg"]=now.year-boroughs["YearBuilt"]
change={2019:0}
boroughs=boroughs.replace(change)
boroughs.head()


# In[313]:


boroughs.groupby(["Borough","ZipCode"])["ageofbldg"].mean()
# boroughs.groupby(["Borough","ZipCode"])["ageofbldg"].describe()


# In[413]:


plt.rcParams["figure.figsize"] = (8, 5)
ax=sns.barplot(x='Borough',y='ageofbldg',data=boroughs)
ax.set_xticklabels(ax.get_xticklabels(),fontsize=11, rotation=0, ha="right")
plt.tight_layout()
plt.show()


# In[ ]:





# ## Question 4 - Predict Complaint Types

# #### Can a predictive model be built for future prediction of the possibility of complaints of the specific type that you identified in response to Question 1?

# ##### MultiLinear Model......THIS IS DONE JUST TO GIVE AN IDEA. IT WILL BE IMPROVED LATER ON.

# In[144]:


df2 = df2.rename(columns={'Agency Name2': 'Agency_Name2', 'Open Data Channel Type2': 'Open_Data_Channel_Type2',
                       "Park Facility Name2":"Park_Facility_Name2",'Park Borough2':'Park_Borough2',
                         'Complaint Type 2':'Complaint_Type_2',"Incident Zip3":"Incident_Zip3"})


# In[368]:


df2.head()


# In[381]:


df3.columns


# In[378]:


# df2.corr()
df3=df2.drop("Park_Borough2",axis=1)


# In[385]:


sns.heatmap(df3.corr(),cmap='coolwarm',annot=True,linewidths=1)


# In[49]:


X=df2.loc[:,df3.columns != "Complaint_Type_2"]
y=df3["Complaint_Type_2"]
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)


# In[50]:


regr = linear_model.LinearRegression()
regr.fit(X, y)


# In[51]:


print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)


# In[ ]:


width = 12
height = 10
plt.figure(figsize=(width, height))
sns.regplot("Park_Facility_Name2", "Complaint_Type_2", df2)
plt.ylim(0,)


# In[ ]:


width = 12
height = 10
plt.figure(figsize=(width, height))
sns.residplot(df2['Park_Facility_Name2'], df2['Complaint_Type_2'])
plt.show()


# In[63]:


Y_hat = regr.predict(X)


# In[105]:


plt.figure(figsize=(width, height))


ax1 = sns.distplot(df3['Complaint_Type_2'], hist=False, color="r", label="Actual Value")
sns.distplot(Y_hat, hist=False, color="b", label="Fitted Values" , ax=ax1)


plt.title('Actual vs Fitted Values')
plt.xlabel('')
plt.ylabel('')

plt.show()
plt.close()


# In[66]:


mse = mean_squared_error(df3['Complaint_Type_2'], Y_hat)
print('The mean square error (MSE) is: ', mse)
rmse=math.sqrt(mse)
print('The root mean square error (RMSE) is: ', rmse)


# In[80]:


#confusion_matrix(y, Y_hat)


# In[100]:


# smf
smf_model=smf.ols("Complaint_Type_2 ~ Agency2+Agency_Name2+Borough2+Open_Data_Channel_Type2+Park_Facility_Name2", df3).fit()
print(smf_model.summary())


# In[98]:


#%%capture
#gather features
#features = "+".join(df2.columns - ["Complaint_Type_2"])

# get y and X dataframes based on this regression:
#y, X = dmatrices('Complaint_Type_2 ~' + features, df2, return_type='dataframe')


# In[99]:


#vif = pd.DataFrame()
#vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
#vif["features"] = X.columns
#vif.round(1)


# In[ ]:





# ##### Decision tree

# In[415]:


dtree = DecisionTreeClassifier()


# In[417]:


dtree.fit(X_trainset,y_trainset)


# In[420]:


predictions = dtree.predict(X_testset)


# In[423]:


print(classification_report(y_testset,predictions))


# In[424]:


print(confusion_matrix(y_testset,predictions))


# In[ ]:





# ##### Random Forest

# In[427]:


# memory error !

rfc = RandomForestClassifier(n_estimators=5)
rfc.fit(X_trainset, y_trainset)


# In[ ]:


rfc_pred = rfc.predict(X_testset)


# In[ ]:


print(classification_report(y_testset,rfc_pred))


# In[ ]:


print(confusion_matrix(y_testset,rfc_pred))


# In[ ]:





# #### Can the model that you developed use Number of Floors in an address as a possible predictive feature?
# 

# In[321]:


boroughs4.columns


# In[387]:


# list(boroughs["NumFloors"].unique())

# number_of_floors can be grouped_by zipcode and cross reference to complaint dataset, 
# then can be used a predictive feature.

# BUT AGAIN, DUE TO MEMORY ERROR, UNABLE TO DO SO.


# In[ ]:


boroughs4["Y"]=boroughs4[""]

X2=df.loc[:,boroughs4.columns != "Y"]
y2=df["Y"]
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X2, y2, test_size=0.3, random_state=3)


# In[ ]:




