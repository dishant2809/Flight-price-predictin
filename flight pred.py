import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, r2_score


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



train = pd.read_excel('Data_Train.xlsx')


pd.set_option('display.max_columns',None)


train.head()


train.isnull().sum()


train.dropna(inplace=True)


train.head(1)


train['Journey_data'] = pd.to_datetime(train['Date_of_Journey'], format='%d/%m/%Y').dt.day

train['Journey_month'] = pd.to_datetime(train['Date_of_Journey'],format='%d/%m/%Y').dt.month



train = train.drop('Date_of_Journey',1)


train['hrs'] = pd.to_datetime(train['Dep_Time']).dt.hour


train['min'] = pd.to_datetime(train['Dep_Time']).dt.minute


train = train.drop('Dep_Time',1)


train.head(2)

duration = list(train["Duration"])
for i in range(len(duration)):
    if len(duration[i].split()) != 2:
        if "h" in duration[i]:
            duration[i] = duration[i].strip() + " 0m"
        else:
            duration[i] = "0h " + duration[i]
            
            
duration_hours = []
duration_mins = []
for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep = "h")[0]))
    duration_mins.append(int(duration[i].split(sep = "m")[0].split()[-1])) 



train['duration_hrs'] = duration_hours
train['duration_min'] = duration_mins



train['duration_min'].value_counts()



train = train.drop('Duration',1)


train['Airline'].value_counts()


Airline = train['Airline']


Airline = pd.get_dummies(Airline,drop_first=True)


Airline.head()


Source = train['Source']
Source = pd.get_dummies(Source,drop_first=True)
Source.head()


Destination = train['Destination']
Destination = pd.get_dummies(Destination,drop_first=True)
Destination.head()


train = train.drop(['Route','Additional_Info'],1)


train['Arrival_hours'] = pd.to_datetime(train['Arrival_Time']).dt.hour
train['Arrival_min'] = pd.to_datetime(train['Arrival_Time']).dt.minute


train = train.drop('Arrival_Time',1)

train.head(2)

train['Total_Stops'] = train['Total_Stops'].replace({'non-stop':0,'1 stop':1,'2 stops':2,'3 stops':3,'4 stops':4})

train = train.drop(['Airline','Destination','Source'],1)


train_df = pd.concat([train,Airline,Source,Destination],1)

train_df.head()


train_df.shape


X = train.drop('Price',1)
y = train['Price']


from sklearn.model_selection import train_test_split
X_train, y_train, X_test, y_test = train_test_split(X,y, random_state=5)


from sklearn.linear_model import LogisticRegression
rcls = LogisticRegression()
rcls.fit(X_train,X_test)

y_pred = rcls.predict(y_train)

r2_score(y_test,y_pred)
