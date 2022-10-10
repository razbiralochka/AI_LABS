import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

tf.config.set_visible_devices([],'GPU')

DataSet=np.genfromtxt("DataSet.txt")


Input = DataSet[:,0]
Output = DataSet[:,1]


WineNET = tf.keras.models.load_model('WineNet')

result = WineNET.predict(Input)




def mapping(x,a,b,c,d):
    res = a*pow(x,3)+b*pow(x,2)+c*x+d
    return res

popt, _ = curve_fit(mapping, Input, Output) 


a, b, c, d = popt 

mapping_res = mapping(Input,a,b,c,d)


plt.plot(Input,Output, label = 'func')
plt.plot(Input,result, label = 'nn_func')
plt.plot(Input,mapping_res, label = 'mapping')
plt.legend(loc=4)
plt.show()