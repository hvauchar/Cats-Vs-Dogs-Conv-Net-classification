import tensorflow as tf 
model = tf.keras.models.load_model('dogsVscats_tensorboard.model')
import matplotlib.pyplot as plt
import cv2
import matplotlib.image as mpimg
import numpy as np
CATOGRTIES = ['Dog','Cat']
def prepare(filepath):
	IMG_SIZE=100
	img_array = cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
	new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
	img=mpimg.imread(filepath)
	imgplot = plt.imshow(img)
	return new_array.reshape(-1,IMG_SIZE, IMG_SIZE, 1)
	

prediction = model.predict([prepare("dog_google.JPG")])
plt.title("Neural Network prediction is {}".format(CATOGRTIES[int(prediction[0][0])]))
plt.show()
print(CATOGRTIES[int(prediction[0][0])])