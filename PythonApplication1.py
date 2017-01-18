import pandas as pd
import quandl,math
import numpy as np
import numpy.linalg as lina 
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression

data = pd.read_csv('data.csv')
data = data[['Id', 'Name', 'Age Category', 'Sex', 'Rank', 'Time', 'Pace', 'Year']]

"""
df = quandl.get('WIKI/GOOGL')
df =df[["Adj. Open", "Adj. High", "Adj. Low", "Adj. Close", "Adj. Volume",]]
df["HL_PCT"] = (df['Adj. High']-df["Adj. Close"])/df["Adj. Close"]*100
df["PCT_Change"] = (df['Adj. Close']-df["Adj. Open"])/df["Adj. Open"]*100
df = df [['Adj. Close', 'HL_PCT', 'PCT_Change', 'Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.01*len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)
"""

#using algorithm embedded
'''
X = np.array(data[['Rank']]) #drop label column and return a new data frame
y = np.array(data['Time']) 
X = preprocessing.scale(X) # normalizes data


X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

clf = LinearRegression()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print accuracy
'''

#formulated linear regression

x = np.array(data[['Rank']])
y = np.array(data['Time']) 
x = preprocessing.scale(x) # normalizes data

X_train, X_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.2)
size = X_train.size
Array1 = np.ones( (size,1) )
X_train =np.append((X_train), (Array1), axis=1)

xt_y=np.dot(X_train.transpose(),y_train)
xt_x=np.dot(X_train.transpose(),X_train)

inv_xt_x=lina.inv(xt_x)
final = np.dot(inv_xt_x,xt_y)
print final
