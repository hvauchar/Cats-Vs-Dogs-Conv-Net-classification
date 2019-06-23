import numpy as np 
import cv2
import os
import matplotlib.pyplot as plt 
DATADIR = "PetImages"
CATEGORIES = ["Dog","Cat"]
IMG_SIZE=100
# new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
# plt.imshow(new_array,cmap="gray")
# plt.show()	
training_data=[]	
def create_training_data():
	
	for category in CATEGORIES:
		path = os.path.join(DATADIR,category)  # path to cats or dogs dir
		class_num = CATEGORIES.index(category)
		for img in os.listdir(path):
			try:
				img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
				new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
				training_data.append([new_array,class_num])
				print("working!")
			except Exception as e:
				pass


create_training_data()
print(len(training_data))

import random 
random.shuffle(training_data)
for sample in training_data[:10]:
	print(sample[1])

X=[]
Y=[]
for fetures, lable in training_data:
 	X.append(fetures)
 	Y.append(lable)
X=np.array(X).reshape(-1,IMG_SIZE,IMG_SIZE,1)
import pickle
pickle_out = open("X.pickle","wb")
pickle.dump(X,pickle_out)
pickle_out.close()
pickle_out = open("Y.pickle","wb")
pickle.dump(Y,pickle_out)
pickle_out.close()
pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)
print(X[1])