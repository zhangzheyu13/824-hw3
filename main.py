import pickle
import numpy as np

import tensorflow as tf
from tensorflow.python.keras.models import Model, load_model, Sequential
from tensorflow.python.keras.layers import Dense, Input, Lambda, Dropout, Reshape, LSTM
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import TensorBoard
from keras.utils.np_utils import to_categorical
import random


feature_dim = 512
num_actions = 51

train_list = pickle.load(open('annotated_train_set.p', 'rb'))['data']
test_list = pickle.load(open('randomized_annotated_test_set_no_name_no_num.p', 'rb'))['data']

random.shuffle(train_list)
train_data = np.stack(tuple([ele['features'] for ele in train_list]), axis=0)

train_labels = np.array([ele['class_num'] for ele in train_list], dtype=np.int32)
train_labels = to_categorical(train_labels, num_classes=num_actions)

test_data = np.stack(tuple([ele['features'] for ele in test_list]), axis=0)


# MLP definition
def _2d23d(x):
	return K.reshape(x, (100, 512))

def _3d22d(x):
	x = K.reshape(x, (10, 10, 51))
	return K.mean(x, axis=1)


input = Input(shape=(10,512))
x = Lambda(_2d23d)(input)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(num_actions, activation='softmax')(x)
x = Lambda(_3d22d)(x)

mlp = Model(input, x)

mlp.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])


# LSTM definition
lstm = Sequential()
lstm.add(LSTM(256, input_shape=(10,512)))
lstm.add(Dense(51, activation='softmax'))

lstm.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])


# train, val, test, visualize
if True:
	tensorboard = TensorBoard(log_dir='./mlp_logs')
	model = mlp
	batch_size = 10
else:
	tensorboard = TensorBoard(log_dir='./lstm_logs')
	model = lstm
	batch_size = 32

model.fit(x=train_data, y=train_labels, batch_size=batch_size, epochs=200, validation_split=0.25, callbacks=[tensorboard])

y_pred = model.predict(test_data, batch_size=batch_size)
results = np.argmax(y_pred, axis=1)
file = open('part1.1.txt', 'w+')
for l in results.tolist():
	file.write("%d\n" % l)
file.close()