# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 18:33:48 2023

@author: gavin
"""

# want at least 8 shots in each rally

file = open(r'C:\gavin\software\python\pytorch\karpathy\makemore\makemore\tennis_shots_new_all_final_reduced.txt','r')
good_points = []

len_of_pt = 6
for idx,line in enumerate(file):
    if len(line.split(',')) >= len_of_pt:
        good_points.append(line)
        
file.close()

file = open(r'C:\gavin\software\python\pytorch\karpathy\makemore\makemore\tennis_shots_new_all_final_reduced_MLM.txt','w')
for point in good_points:
	#file.write(item+",")
    # write each element in the list - which is a list
    # as a line in the file. Each list is joined by comma
    # and written out
    #file.write(point.replace(',',' '))
    file.write(point)
    #file.write(point + "\n")

file.close()  

file = open(r'C:\gavin\software\python\pytorch\karpathy\makemore\makemore\tennis_shots_new_all_final_reduced_MLM.txt','r')
good_points = []

for idx,line in enumerate(file):
        good_points.append(line.strip())
        
file.close()


        
from tensorflow.keras.layers import TextVectorization    

vectorize_layer = TextVectorization(output_mode="int")

vectorize_layer.adapt(good_points)
vocab = vectorize_layer.get_vocabulary()
len(vocab)
vocab[:10]
vocab[-10:]

vectorize_layer(good_points[0])
vectorize_layer(good_points)


vocab_new = vocab[2 : len(vocab) - 1] + ["[mask]"]

vectorize_layer.set_vocabulary(vocab_new)
vectorize_layer(good_points)

# different lengths??
len(vocab_new)
len(vectorize_layer.get_vocabulary())


# note 
features = {
        "token_ids": vectorize_layer(good_points),
        "mask_positions": ["mask_positions"],
    }


