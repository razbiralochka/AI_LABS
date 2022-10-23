import re
import numpy as np
from collections import Counter
import pickle
path = "train_text.txt"
text_file = open(path, "r", encoding="utf-8")






word_lists = [
    [word for word in re.split("[^a-z]", line.lower()) if word]
    for line in text_file.readlines()
]  # [line][word]
text_file.close()




set_ = set(word for line in word_lists for word in line)
dict_ = dict((word, i) for i, word in enumerate(set_))

N = round(len(word_lists))

nn_words=list()


train_ds = np.zeros([N,200])

def wfile(i):
    for word in word_lists[i]:
        
            if (word in dict_):
                nn_words.append((dict_[word]))
            else:
                nn_words.append(-1)
            
       
    print(word_lists[i])
    print(sorted(word_lists[i]))
    print(nn_words)
    for j, var in enumerate (nn_words):
        train_ds[i,j] = nn_words[j]
    
    nn_words.clear()


S=0

for i in range(N):
    print("____________________________________")
    print(wfile(i))
    S=i
    

print("__________________")

print(dict_)

with open('dictf.pkl', 'wb') as f:
    pickle.dump(dict_, f)

train_ds=np.round(train_ds)


np.savetxt("text_in_nums.txt", train_ds, delimiter=" ")