import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt

tf.config.set_visible_devices([],'GPU')

DataSet=np.genfromtxt("DataSet.txt")


Input = DataSet[:,0]
Output = DataSet[:,1]

model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(10, activation='sigmoid',input_dim=1),
  tf.keras.layers.Dense(300, activation='relu'),
  tf.keras.layers.Dense(10, activation='sigmoid'),
  tf.keras.layers.Dense(5, activation='relu'),
  tf.keras.layers.Dense(1)
])






model.summary()


model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=1e-3), loss='mse', metrics=['acc'])


model.fit(Input, Output, epochs=1000,batch_size = 10)

model.summary()

model.save('WineNet')



