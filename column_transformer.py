import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder


df = pd.read_csv('./covid_toy.csv')
print(df.head())
print(df.isnull().sum())

print(df['cough'].value_counts())
print(df['fever'].value_counts())
print(df['city'].value_counts())

X_train,X_test,y_train,y_test = train_test_split(df.drop(columns=['has_covid']),df['has_covid'], test_size=0.2)

transformer = ColumnTransformer(transformers=[
    ('tnf1',SimpleImputer(),['fever']),
    ('tnf2',OrdinalEncoder(categories=[['Mild','Strong']]),['cough']),
    ('tnf3',OneHotEncoder(sparse_output=False,drop='first'),['gender','city'])
],remainder='passthrough')
'''simpleImputer: to fill the missing data by mean in fever'''

print(transformer.fit_transform(X_train).shape) # 100 columnd: 80 train and 20 test
print(transformer.transform(X_test).shape)