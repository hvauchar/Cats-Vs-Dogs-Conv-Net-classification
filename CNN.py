import tensorflow as tf 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.python.keras.callbacks import TensorBoard
import pickle, time

NAME = "CatsVsDogs--46x2-{}".format(int(time.time()))

tensorboard =TensorBoard(log_dir="logs/{}".format(NAME))

X=pickle.load(open("X.pickle","rb"))
y=pickle.load(open("Y.pickle","rb"))

X = X/255.0

model = Sequential()
model.add(Conv2D(64, (3,3), input_shape=X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(64))
model.add(Activation("relu"))
model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])
model.fit(X,y,batch_size=32,epochs=5,validation_split=0.3,callbacks=[tensorboard])
model.save('dogsVscats_tensorboard.model')
#tensorboard --logdir=logs/