import pandas as pd
import statsmodels.api as sm

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

hawaii_data = pd.read_csv('Honolulu.csv')
hawaii_trimmed = hawaii_data[['YEAR', 'MONTH', 'DAY', 'PRCP', 'TAVG', 'TMAX', 'TMIN']]

san_diego_data = pd.read_csv('San Diego.csv')
san_diego_trimmed = san_diego_data[['YEAR', 'MONTH', 'DAY', 'PRCP', 'TMAX', 'TMIN']]

seattle_data = pd.read_csv('Seattle.csv')
seattle_trimmed = seattle_data[['YEAR', 'MONTH', 'DAY', 'PRCP', 'TMAX', 'TMIN']]


# trim hawaii to relevant time frame
is_year = hawaii_trimmed.YEAR.isin(range(2000,2020))
hawaii_trimmed = hawaii_trimmed[is_year]

is_march = hawaii_trimmed['MONTH'] == 3
hawaii_trimmed = hawaii_trimmed[is_march]


# trim san diego to relevant time frame
is_year = san_diego_trimmed.YEAR.isin(range(2000,2020))
san_diego_trimmed = san_diego_trimmed[is_year]

is_march = san_diego_trimmed['MONTH'] == 3
san_diego_trimmed = san_diego_trimmed[is_march]

# trim seattle to relevant time frame
is_year = seattle_trimmed.YEAR.isin(range(2000,2020))
seattle_trimmed = seattle_trimmed[is_year]

is_march = seattle_trimmed['MONTH'] == 3
seattle_trimmed = seattle_trimmed[is_march]


# linear regression, hawaii
X = hawaii_trimmed[['YEAR', 'DAY']]
X = sm.add_constant(X)
y_prcp = hawaii_trimmed[['PRCP']]

lr_model_prcp = sm.OLS(y_prcp, X).fit()
print(lr_model_prcp.summary())


# linear regression, san diego
X = san_diego_trimmed[['YEAR', 'DAY']]
X = sm.add_constant(X)
y_prcp = san_diego_trimmed[['PRCP']]

lr_model_prcp = sm.OLS(y_prcp, X).fit()
print(lr_model_prcp.summary())


# linear regression, seattle
X = seattle_trimmed[['YEAR', 'DAY']]
X = sm.add_constant(X)
y_prcp = seattle_trimmed[['PRCP']]


lr_model_prcp = sm.OLS(y_prcp, X).fit()
print(lr_model_prcp.summary())


# logistic regression, precipitation, hawaii
hawaii_log = hawaii_trimmed

hawaii_log['PRCP'] = np.where(hawaii_log['PRCP'] > 0, 1, 0)

X = hawaii_log[['YEAR', 'DAY']]
y = hawaii_log['PRCP']

average = 0

for i in range(100):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    logmodel = LogisticRegression()
    logmodel.fit(X_train, y_train)
    
    predictions = logmodel.predict(X_test)
    average += accuracy_score(y_test, predictions.ravel())
    
print(average/100)

date1 = pd.DataFrame(index=np.arange(1), columns=np.arange(2))
date1.loc[0] = [2021, 13]

date2 = pd.DataFrame(index=np.arange(1), columns=np.arange(2))
date2.loc[0] = [2021, 14]

log_3_13_hawaii = logmodel.predict(date1)
log_3_13_hawaii = logmodel.predict(date2)

# logistic regression, precipitation, san diego
diego_log = san_diego_trimmed

diego_log['PRCP'] = np.where(diego_log['PRCP'] > 0, 1, 0)

X = diego_log[['YEAR', 'DAY']]
y = diego_log['PRCP']

average = 0

for i in range(100):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    logmodel = LogisticRegression()
    logmodel.fit(X_train, y_train)
    
    predictions = logmodel.predict(X_test)
    average += accuracy_score(y_test, predictions.ravel())
    
print(average/100)

log_3_13_diego = logmodel.predict(date1)
log_3_14_diego = logmodel.predict(date2)

# logistic regression, precipitation, seattle
seattle_log = seattle_trimmed

seattle_log['PRCP'] = np.where(seattle_log['PRCP'] > 0, 1, 0)

X = seattle_log[['YEAR', 'DAY']]
y = seattle_log['PRCP']

average = 0

for i in range(100):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    logmodel = LogisticRegression()
    logmodel.fit(X_train, y_train)
    
    predictions = logmodel.predict(X_test)
    average += accuracy_score(y_test, predictions.ravel())
print(average/100)

log_3_13_seattle = logmodel.predict(date1)
log_3_14_seattle = logmodel.predict(date2)


# knn, precipitation, hawaii
X = hawaii_log[['YEAR', 'DAY']]
y = hawaii_log['PRCP']

total = 0

for i in range(100): 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train,y_train)
    predictions = knn.predict(X_test)
    total += accuracy_score(y_test,predictions)
    
print(total/100)

knn_3_13_hawaii = knn.predict(date1)
knn_3_14_hawaii = knn.predict(date2)

# knn, precipitation, san diego
X = diego_log[['YEAR', 'DAY']]
y = diego_log['PRCP']

total = 0

for i in range(100): 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train,y_train)
    predictions = knn.predict(X_test)
    total += accuracy_score(y_test,predictions)
    
print(total/100)

knn_3_13_diego = knn.predict(date1)
knn_3_14_diego = knn.predict(date2)

# knn, precipitation, seattle
X = seattle_log[['YEAR', 'DAY']]
y = seattle_log['PRCP']

total = 0

for i in range(100): 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train,y_train)
    predictions = knn.predict(X_test)
    total += accuracy_score(y_test,predictions)
    
print(total/100)

knn_3_13_seattle = knn.predict(date1)
knn_3_14_seattle = knn.predict(date2)

# random forest, precipitation, hawaii
X = hawaii_log[['YEAR', 'DAY']]
y = hawaii_log['PRCP']

total = 0

for i in range(100): 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    rfc = RandomForestClassifier(n_estimators=100)
    rfc.fit(X_train, y_train.values.ravel())
    predictions = rfc.predict(X_test)
    total += accuracy_score(y_test,predictions)

print(total/100)

rf_3_13_hawaii = rfc.predict(date1)
rf_3_14_hawaii = rfc.predict(date2)

# random forest, precipitation, san diego
X = diego_log[['YEAR', 'DAY']]
y = diego_log['PRCP']

total = 0

for i in range(100): 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    rfc = RandomForestClassifier(n_estimators=100)
    rfc.fit(X_train, y_train.values.ravel())
    predictions = rfc.predict(X_test)
    total += accuracy_score(y_test,predictions)

print(total/100)

rf_3_13_diego = rfc.predict(date1)
rf_3_14_diego = rfc.predict(date2)


# random forest, precipitation, seattle
X = seattle_log[['YEAR', 'DAY']]
y = seattle_log['PRCP']

total = 0

for i in range(100): 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    rfc = RandomForestClassifier(n_estimators=100)
    rfc.fit(X_train, y_train.values.ravel())
    predictions = rfc.predict(X_test)
    total += accuracy_score(y_test,predictions)

print(total/100)

rf_3_13_seattle = rfc.predict(date1)
rf_3_14_seattle = rfc.predict(date2)



## ITERATIVE MODEL
#Hawaii
hawaii_it = hawaii_data[['YEAR', 'MONTH', 'DAY', 'PRCP', 'TMAX', 'TMIN']]
is_year = hawaii_it.YEAR.isin(range(2000,2021))
hawaii_it = hawaii_it[is_year]
hawaii_it['PRCP'] = np.where(hawaii_it['PRCP'] > 0, 1, 0)


#San Diego
diego_it = san_diego_data[['YEAR', 'MONTH', 'DAY', 'PRCP', 'TMAX', 'TMIN']]
is_year = diego_it.YEAR.isin(range(2000,2021))
diego_it = diego_it[is_year]

#Seattle
seattle_it = seattle_data[['YEAR', 'MONTH', 'DAY', 'PRCP', 'TMAX', 'TMIN']]
is_year = seattle_it.YEAR.isin(range(2000,2021))
seattle_it = seattle_it[is_year]

# Setting the number of days to look back on
days_back = 5

# Variables to look back on
var_list = ['TMAX', 'TMIN', 'PRCP']


## Hawaii
# Creating new columns with previous weather information
for i in range(days_back):
    new_colnames = [j+'_'+str(i+1)+'_DAY' for j in var_list]
    hawaii_it[new_colnames] = hawaii_it[var_list].shift(i+1)
 
 #trim to march
is_march = hawaii_it['MONTH'] == 3
diego_it = hawaii_it[is_march]   
 
    # Dropping all NaN values and reseting the indices of the dataframe
hawaii_it.dropna(inplace=True)
hawaii_it.reset_index(drop=True, inplace=True)

# Establishing predictor variables for model
X = hawaii_it[['YEAR', 'DAY', 'TMAX_1_DAY', 'TMIN_1_DAY', 'PRCP_1_DAY',
         'TMAX_2_DAY', 'TMIN_2_DAY', 'PRCP_2_DAY',
         'TMAX_3_DAY', 'TMIN_3_DAY', 'PRCP_3_DAY',
         'TMAX_4_DAY', 'TMIN_4_DAY', 'PRCP_4_DAY',
         'TMAX_5_DAY', 'TMIN_5_DAY', 'PRCP_5_DAY']]

# Establishing responding variables
y = hawaii_it[['PRCP']]

average = 0

for i in range(100):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    logmodel = LogisticRegression()
    logmodel.fit(X_train, y_train)
    
    predictions = logmodel.predict(X_test)
    average += accuracy_score(y_test, predictions.ravel())
    
print(average/100)


## San Diego
# Creating new columns with previous weather information
for i in range(days_back):
    new_colnames = [j+'_'+str(i+1)+'_DAY' for j in var_list]
    diego_it[new_colnames] = diego_it[var_list].shift(i+1)

#trim to march
is_march = diego_it['MONTH'] == 3
diego_it = diego_it[is_march]

    # Dropping all NaN values and reseting the indices of the dataframe
diego_it.dropna(inplace=True)
diego_it.reset_index(drop=True, inplace=True)

# Establishing predictor variables for model
X = diego_it[['YEAR', 'DAY', 'TMAX_1_DAY', 'TMIN_1_DAY', 'PRCP_1_DAY',
         'TMAX_2_DAY', 'TMIN_2_DAY', 'PRCP_2_DAY',
         'TMAX_3_DAY', 'TMIN_3_DAY', 'PRCP_3_DAY',
         'TMAX_4_DAY', 'TMIN_4_DAY', 'PRCP_4_DAY',
         'TMAX_5_DAY', 'TMIN_5_DAY', 'PRCP_5_DAY']]

# Establishing responding variables
y = diego_it[['PRCP']]

average = 0

for i in range(100):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    logmodel = LogisticRegression()
    logmodel.fit(X_train, y_train)
    
    predictions = logmodel.predict(X_test)
    average += accuracy_score(y_test, predictions.ravel())
    
print(average/100)

## Seattle
# Creating new columns with previous weather information
for i in range(days_back):
    new_colnames = [j+'_'+str(i+1)+'_DAY' for j in var_list]
    seattle_it[new_colnames] = seattle_it[var_list].shift(i+1)
    
 #trim to march
is_march = seattle_it['MONTH'] == 3
seattle_it = seattle_it[is_march]       

    # Dropping all NaN values and reseting the indices of the dataframe
seattle_it.dropna(inplace=True)
seattle_it.reset_index(drop=True, inplace=True)

# Establishing predictor variables for model
X = seattle_it[['YEAR', 'DAY', 'TMAX_1_DAY', 'TMIN_1_DAY', 'PRCP_1_DAY',
         'TMAX_2_DAY', 'TMIN_2_DAY', 'PRCP_2_DAY',
         'TMAX_3_DAY', 'TMIN_3_DAY', 'PRCP_3_DAY',
         'TMAX_4_DAY', 'TMIN_4_DAY', 'PRCP_4_DAY',
         'TMAX_5_DAY', 'TMIN_5_DAY', 'PRCP_5_DAY']]

# Establishing responding variables
y = seattle_it[['PRCP']]

average = 0

for i in range(100):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    logmodel = LogisticRegression()
    logmodel.fit(X_train, y_train)
    
    predictions = logmodel.predict(X_test)
    average += accuracy_score(y_test, predictions.ravel())
    
print(average/100)