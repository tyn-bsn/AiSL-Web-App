import pickle
import tensorflowjs as tfjs
import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

data_dict = pickle.load(open('./data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])
from sklearn.preprocessing import LabelEncoder

lb = LabelEncoder()
labels = lb.fit_transform(labels).astype(int)
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)
y_train = y_train.astype(int)
y_test = y_test.astype(int)
model = keras.models.Sequential()
opt = keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss="sparse_categorical_crossentropy", optimizer=opt)
model.fit(x_train, y_train, epochs=10)  # You might need to adjust the number of epochs.

y_predict = model.predict(x_test).argmax(axis=1)

score = accuracy_score(y_test, y_predict)

print('{}% of samples were classified correctly!'.format(score * 100))

import os
directory = os.getcwd()
tfjs.converters.save_keras_model(model, directory)