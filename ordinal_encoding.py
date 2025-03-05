import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('./customer.csv')
df = df.iloc[:,2:]
print(df.head())

X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, 0:2], df.iloc[:,-1], test_size=0.2)

print(X_train.head())

# Ordinal Encoder class object
oe = OrdinalEncoder(categories=[['Poor','Average','Good'],['School','UG','PG']])

oe.fit(X_train)
X_train = oe.transform(X_train)
X_test = oe.transform(X_test)

print(oe.categories_)
print(X_train)

# Lable Encoding on Purchased (o/p column)
le = LabelEncoder()

le.fit(y_train)
le.classes_

y_train = le.transform(y_train)
y_test = le.transform(y_test)

print(y_train)