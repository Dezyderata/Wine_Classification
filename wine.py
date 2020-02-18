#!/usr/bin/env python3

import sqlite3
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

try:
    sqliteConnection = sqlite3.connect('wine')
    print('DB connection successful')
    cursor = sqliteConnection.cursor()
    select_que = 'SELECT * FROM wine_dataset'
    cursor.execute(select_que)
    my_wine_data = cursor.fetchall()
    cursor.close()
except sqlite3.Error as error:
    print(f'Connection fail: {error}')
finally:
    if sqliteConnection:
        sqliteConnection.close()
        print('Connection closed')
#print(my_wine_data)
print(type(my_wine_data))
columns_names = ['Alcohol', 'Malic_acid', 'Ash', 'Alca_of_ash', 'Magnesium', 'Total_phenols', 'Flavanoids', 'Nonfav_phenols', 'Proanthocyanins', 'color_int', 'Hue', 'OD280/OD315', 'Proline', 'Wine_type']
wine_df = pd.DataFrame(my_wine_data, columns=columns_names)
for item in wine_df.columns[:-1]:
    wine_df[item][1:] = pd.to_numeric(wine_df[item][1:])
print(wine_df.head())
X = wine_df.iloc[:,:-1]
y = wine_df.iloc[:,-1:]
print(X.head())
print(y.head())

my_scaler = StandardScaler()
X[:][1:] = my_scaler.fit_transform(X[:][1:])
print(X.head())
