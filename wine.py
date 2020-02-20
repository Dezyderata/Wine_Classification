#!/usr/bin/env python3

import sqlite3
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split



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

wine_df = pd.DataFrame(my_wine_data)
for item in wine_df.columns[:-1]:
    wine_df[item][1:] = pd.to_numeric(wine_df[item][1:])

X = wine_df.iloc[:,:-1]
y = wine_df.iloc[:,-1:]
my_scaler = StandardScaler()
X[:][1:] = my_scaler.fit_transform(X[:][1:])
y_encoder = LabelEncoder()
y = y_encoder.fit_transform(y[1:].values.ravel())
new_x = X.values

X_train, X_test, y_train, y_test = train_test_split(new_x[1:, :], y, test_size=0.2, random_state=5)
y_test_arr = y_test
X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
y_train = tf.convert_to_tensor(y_train, dtype=tf.int8)
y_test = tf.convert_to_tensor(y_test, dtype=tf.int8)



model = keras.Sequential([
        keras.layers.Dense(26, activation='relu', input_shape=(13,)),
        keras.layers.Dense(13, activation='relu'),
        keras.layers.Dense(3, activation='softmax')
])


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=15)

test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)

print('\nTest accuracy:', test_acc)

predictions = model.predict(X_test)
y_test_decoded = y_encoder.inverse_transform(y_test_arr)

for i in range(len(predictions)):
    print(f'Prediction: {y_encoder.inverse_transform([np.argmax(predictions[i])])}, real value: {y_test_decoded[i]}')
