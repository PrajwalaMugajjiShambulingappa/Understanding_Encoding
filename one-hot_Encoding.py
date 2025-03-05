import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


df = pd.read_csv('./cars.csv')

print(df.head())
print(df['brand'].value_counts())
print(df['fuel'].value_counts())
print(df['owner'].value_counts())

# OneHotEncoding using Pandas
pd.get_dummies(df,columns=['fuel','owner'])

# K-1 OneHotEncoding (using pandas but to avoid multicollinarity)
pd.get_dummies(df,columns=['fuel','owner'],drop_first=True)

# OneHotEncoding using Sklearn
X_train,X_test,y_train,y_test = train_test_split(df.iloc[:,0:4],df.iloc[:,-1],test_size=0.2,random_state=2)
print(X_train.head())

ohe = OneHotEncoder(drop='first',sparse_output=False,dtype=np.int32)

X_train_new = ohe.fit_transform(X_train[['fuel','owner']])
X_test_new = ohe.transform(X_test[['fuel','owner']])

# To sawp the 'brand' and 'km_driven" with the newly encoded data -> won't do it much because we use common tranform instead
np.hstack((X_train[['brand','km_driven']].values,X_train_new))

# OneHotEncoding with Top Categories -> to only have top freq and other brands with less freq into uncommon
counts = df['brand'].value_counts()
df['brand'].nunique()

''' Using this threshold value to filter the brands with top freq'''
threshold = 100

repl = counts[counts <= threshold].index
pd.get_dummies(df['brand'].replace(repl, 'uncommon')).sample(5)