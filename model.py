from keras import backend as K
from keras.layers import Embedding, Lambda, Dense, Input
from keras.models import Sequential, Model
import numpy as np

VOCABULARY_SIZE = 100
EMB_SIZE = 2
INPUT_LENGTH = 5


def sequece():
  model = Sequential()
  model.add(Embedding(VOCABULARY_SIZE, EMB_SIZE, input_length=INPUT_LENGTH))
  model.add(Lambda(lambda x: K.sum(x, axis=1)))
  model.add(Dense(1024, activation='relu'))
  model.add(Dense(512, activation='relu'))
  model.add(Dense(256, activation='relu'))
  model.add(Dense(256, activation='softmax'))

  model.compile(optimizer='rmsprop',
                loss='mse',
                metrics=['accuracy'])
  return model


def functional():
  inputs = Input(shape=(5,))
  x = Embedding(VOCABULARY_SIZE, EMB_SIZE, input_length=INPUT_LENGTH)(inputs)
  x = Lambda(lambda x: K.sum(x, axis=1))(x)
  x = Dense(1024, activation='relu')(x)
  x = Dense(512, activation='relu')(x)
  x = Dense(256, activation='relu')(x)
  predictions = Dense(256, activation='softmax')(x)
  functional_model = Model(inputs=inputs, outputs=predictions)
  functional_model.compile(optimizer='rmsprop',
                           loss='mse',
                           metrics=['accuracy'])
  return functional_model


x = np.random.random(5000).reshape((1000, 5))
y = np.random.random(256000).reshape((1000, 256))
model = functional()
model.summary()
model.fit(x, y)

output = model.predict(np.random.random(5).reshape((1, 5)))
print(output)
