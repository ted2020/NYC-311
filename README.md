# NYC 311
# NYC 311 Complaint Type and Housing Characteristics

### 12 GB of data
#### 311 complaint dataset
##### This dataset is available at https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2010-to-Present/erm2-nwe9. You can download part of this data by using SODA API.


### 2GB of data
#### PLUTO dataset for housing
##### This dataset for housing can be accessed from https://data.cityofnewyork.us/City-Government/Primary-Land-Use-Tax-Lot-Output-PLUTO-/xuk2-nczf


#### import requests
#### import numpy as np
#### import pandas as pd
#### from datetime import datetime
#### from datetime import date
#### import csv
#### import urllib.request as urllib2
#### import dask.dataframe as dd
#### import glob
#### import os
#### from sklearn.model_selection import train_test_split
#### from sklearn.tree import DecisionTreeClassifier
#### from sklearn import linear_model
#### from sklearn.linear_model import LinearRegression
#### from sklearn.metrics import mean_squared_error
#### from sklearn.metrics import confusion_matrix
#### from sklearn.preprocessing import StandardScaler
#### from sklearn.ensemble import RandomForestClassifier
#### from sklearn.metrics import classification_report,confusion_matrix
#### from sklearn import metrics
#### import statsmodels.api as sm
#### import ijson
#### import time
#### import matplotlib.pyplot as plt
#### %matplotlib inline 
#### import seaborn as sns
#### import re
#### import math
#### import statsmodels.formula.api as smf
#### from statsmodels.stats.outliers_influence import variance_inflation_factor
#### from patsy import dmatrices

#### Question 1 - Which type of complaints The Department of Housing Preservation and Development of New York City should focus first?
#### Which type of complaint should the Department of Housing Preservation and Development of New York City focus on first?
#### What is the total number of General Construction complaints that came to the Department of Housing Preservation and Development of New York City?
#### For the Complaint Type that you selected, can you determine the total number of complaints at the Street level?
#### Which approach do you think can be used to find the total number of complaints for an address?
#### Which is the Complaint Type that the Department of Housing Preservation and Development of New York City should address first considering complaints created till 31st Dec 2018 ?
#### Question 2 - What Areas Should the Agency Focus On?
#### Should the Department of Housing Preservation and Development of New York City focus on any particular set of boroughs, ZIP codes, or street (where the complaints are severe) for the specific type of complaints you identified in response to Question 1?
#### Which approach do you think can be used to find the total number of complaints for an address?
#### where homelessness is a problem?
#### Question 3 - What Is the Relationship between Housing Characteristics and Complaints?
#### Does the Complaint Type that you identified in response to Question 1 have an obvious relationship with any particular characteristic or characteristic of the Houses?
#### Can you determine the age of the building from the PLUTO dataset?
#### Question 4 - Predict Complaint Types
#### Can a predictive model be built for future prediction of the possibility of complaints of the specific type that you identified in response to Question 1?
#### Decision Tree
#### Random Forest
#### Can the model that you developed use Number of Floors in an address as a possible predictive feature?