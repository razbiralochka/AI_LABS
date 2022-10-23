import numpy as np
import tensorflow as tf
import re
import pickle
import matplotlib.pyplot as plt
tf.config.set_visible_devices([],'GPU')


class paraNetClass():
    def __init__(self):
        with open('dictf.pkl', 'rb') as f:
            self.dict_of_words = pickle.load(f)
        self.dictThemes = {0:':Кошки (животные);',
                           1:'UNIX-утилита cat для вывода содержимого файлов;',
                           2:'Версии операционной системы OS X, названные в честь семейства кошачьих.',}
        self.a = np.zeros([1,200])
        self.paraNET = tf.keras.models.load_model('ParaNet')
        self.nn_words=list()
    def text_analyse(self,text):
        text = re.split("[^a-z]", text.lower()) 



        for word in text:
                
            if (word in self.dict_of_words):
                self.nn_words.append(self.dict_of_words[word])
                    



        for j, var in enumerate (self.nn_words):
            self.a[0,j] = self.nn_words[j]

        self.nn_words.clear()
        b=self.a.reshape(1,1,200)

        themes = self.paraNET.predict(b)

        #print(themes)

        arg = np.argmax(themes)

        theme = self.dictThemes[arg]


        return themes[0,0]
        


paraNET = paraNetClass()




str = 'A common interactive use of cat for a single file is to output the content of a file to standard output.'



result = paraNET.text_analyse(str[0])
   









