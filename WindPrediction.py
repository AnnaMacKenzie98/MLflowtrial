#!/usr/bin/env python
# coding: utf-8

# In[1]:


###################################################
#import packages

import pandas as pd
#get_ipython().system('pip install influxdb')
import warnings
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import mlflow
import mlflow.sklearn
import logging


logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

from sklearn.neighbors import KNeighborsRegressor
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse

le = preprocessing.LabelEncoder()


# In[2]:


#I do not use this in my code, but it is from example

def eval_metrics(actual, pred):
     rmse = np.sqrt(mean_squared_error(actual, pred))
     mae = mean_absolute_error(actual, pred)
     r2 = r2_score(actual, pred)
     return rmse, mae, r2


# In[4]:

mlflow.set_experiment(experiment_name='Wind Prediction 3')

  ############################Getting the data####################################################################

from influxdb import InfluxDBClient # install via "pip install influxdb"

client = InfluxDBClient(host='influxus.itu.dk', port=8086, username='lsda', password='icanonlyread')
client.switch_database('orkney')

def get_df(results):
     values = results.raw["series"][0]["values"]
     columns = results.raw["series"][0]["columns"]
     df = pd.DataFrame(values, columns=columns).set_index("time")
     df.index = pd.to_datetime(df.index) # Convert to datetime-index
     return df

# Get the last 90 days of power generation data
generation = client.query(
    "SELECT * FROM Generation where time > now()-90d"
    ) # Query written in InfluxQL

# Get the last 90 days of weather forecasts with the shortest lead time
wind  = client.query(
    "SELECT * FROM MetForecasts where time > now()-90d and time <= now() and Lead_hours = '1'"
    ) # Query written in InfluxQL


gen_df = get_df(generation)  #you only need the time and the total columns not ANM or non-anm
wind_df = get_df(wind)


#printing generation df
gen_df=pd.DataFrame(gen_df['Total'])
print('GENERATION DF')
print(gen_df)

#printing wind df
wind_df= wind_df[['Direction', 'Speed']]
wind_df.head()
print("WING DF")
print(wind_df)

mergedata=pd.merge(wind_df, gen_df, on='time', how='inner') #here the time stamps are aligned which gets rid of values that do not have a match- need to justify this

print("GENERATION AND WIND DF MERGED")
print(mergedata)

############################all of below code is just used to ensure data quality####################################

#mergedata.info()
#pd.isnull(mergedata).values.sum()
#mergedata.isin( [0]).sum ()
#can confirm that there are no missing values (the inner join got rid of them)(the original wind df had 673 columns and in the merged there is 671. 
#total had 129454 rows so many of those have been removed )

X=mergedata.iloc[:,[0,1]]
y=mergedata.iloc[:, [2]]
print("y", y)
print("X", X)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4,shuffle=False, stratify=None)

scaler = MinMaxScaler() # scale
#scalerr = ColumnTransformer([("scaler", scaler, ["Speed", "Total"])], remainder="passthrough")# Column Transformer used to apply one-hot-encoding only to col. "Direction"

enc = OneHotEncoder(handle_unknown='ignore', sparse=False) # One-hot-encoder
onehot = ColumnTransformer([("encoder transformer", enc, ["Direction"])], remainder="passthrough")# Column Transformer used to apply one-hot-encoding only to col. "Direction"

ONEHOTENCODING=pd.DataFrame(onehot.fit_transform(x_train), dtype=int)
print(ONEHOTENCODING)

#parameters
num_neighbours=6

#Models

LRmodel=LinearRegression()
KNNR= KNeighborsRegressor(n_neighbors=num_neighbours)

from sklearn.pipeline import Pipeline

# Align the data frames

#LR
pipelineLR = Pipeline(steps=[("encode",onehot ),("scaler", scaler),("Linear Regression Model", LRmodel)])

#KNNR
pipelineKNN = Pipeline(steps=[("encode",onehot ),("scaler", scaler),("KNNR Model", KNNR)])

#LR
pipelineLR_model = pipelineLR.fit(x_train, y_train)
y_pred = pipelineLR_model.predict(x_test)
y_test = y_test.reset_index(drop=True)
print(y_pred)

plt.plot(y_test, label = "y_test")
plt.plot(y_pred, label = "y_pred")
plt.legend()
plt.show()

MSE= mean_squared_error(y_test, y_pred)
MAE=mean_absolute_error(y_test, y_pred)
print(
' mean_squared_error LR : ', MSE)
print(
' mean_absolute_error LR : ', MAE)

#KNNR
pipelineKNN_model = pipelineKNN.fit(x_train, y_train)
y_predKNN = pipelineKNN_model.predict(x_test)
y_testKNN = y_test.reset_index(drop=True)
print(y_predKNN)

plt.plot(y_testKNN, label = "y_test")
plt.plot(y_predKNN, label = "y_pred")
plt.legend()
plt.show()

MSE_KNN= mean_squared_error(y_testKNN, y_predKNN)
MAE_KNN=mean_absolute_error(y_testKNN, y_predKNN)
print(
' mean_squared_error KNN : ', MSE_KNN)
print(
' mean_absolute_error KNN : ', MAE_KNN)

#     #things to log in MLFLOW for LR
# mlflow.log_metric("LR Model MSE", MSE)
# mlflow.log_metric("LR Model MAE", MAE)
# mlflow.sklearn.log_model(pipelineLR_model, "LR model")    
# mlflow.sklearn.save_model(pipelineLR_model, "BDM 3 LR model")

#things to log in MLFLOW for KNN
mlflow.log_param("number of neighbours", num_neighbours)
mlflow.log_metric("KNN Model MSE", MSE_KNN)
mlflow.log_metric("KNN Model MAE", MAE_KNN)
mlflow.sklearn.log_model(pipelineKNN_model, " Wind Prediction KNN Model")

mlflow.sklearn.save_model(pipelineKNN_model, "Wind Prediction 3 KNN model")


# In[ ]:


####################################################
## Do forecasting with the best one

# Get all future forecasts regardless of lead time
forecasts  = client.query(
    "SELECT * FROM MetForecasts where time > now()"
    ) # Query written in InfluxQL
for_df = get_df(forecasts)

# Limit to only the newest source time
newest_source_time = for_df["Source_time"].max()
newest_forecasts = for_df.loc[for_df["Source_time"] == newest_source_time].copy()

# Preprocess the forecasts and do predictions in one fell swoop 
# using your best pipeline.
pipeline.predict(newest_forecasts)


# In[5]:


mlflow.sklearn.save_model(pipelineLR_model, "Wind Prediction 3 KNN model")


# In[ ]:





# In[1]:


#get_ipython().system(' pip freeze > requirement.txt')


# In[ ]:

if __name__ == "__main__":
    main()



# In[ ]:




