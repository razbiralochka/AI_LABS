import numpy as np
import tensorflow as tf
import pickle

with open('dictf.pkl', 'rb') as f:
    dict_ = pickle.load(f)




train = np.genfromtxt('text_in_nums.txt')

themes = np.genfromtxt('Themes.txt')



tf.config.set_visible_devices([],'GPU')



train_y = np.zeros([len(themes),3])



for i, var in enumerate(themes):
    
    train_y[i,int(var)] = 1


print(train_y.shape)

train = train.reshape(len(themes),1,200)
train_y = train_y.reshape(len(themes),1,3)
print(train_y.shape)

paraNET = tf.keras.models.Sequential([
  tf.keras.layers.Conv1D(filters=10,
                         input_shape=(1,200),
                         kernel_size=3,
                         activation='relu',
                         padding = 'same'),
  tf.keras.layers.Dense(500, activation='sigmoid'),
  tf.keras.layers.Dense(100, activation='relu'),
  tf.keras.layers.Dense(30, activation='sigmoid'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(3, activation='softmax')
  
])

paraNET.summary()

paraNET.compile(loss="binary_crossentropy", optimizer="adam")



paraNET.fit(train, train_y, epochs = 500, batch_size=5)



dictThemes = {0:'Спам',
              1:'Запись',
              2:'Общая информация',
              3:'Информация курсантам',
              4:'Тандемы',}


paraNET.save('ParaNet')