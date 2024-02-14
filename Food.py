

import cv2 
from glob import glob
import pathlib
import numpy as np
import pandas as pd
import tensorflow as tf 
import matplotlib.pyplot as plt

# from google.colab import drive
# drive.mount('/content/drive')

from tensorflow.keras import Sequential 
from tensorflow.keras.layers import Flatten,Dense,Conv3D,MaxPool3D #cnn layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

food_path=pathlib.Path(r"C:\Users\user\Desktop\Food\training")

A=list(food_path.glob("Bread/*.jpg"))          # *-shows all #(malignant/*)- shows all files inside the malignant folder
B=list(food_path.glob("Dairyproduct/*.jpg"))
C=list(food_path.glob("Dessert/*.jpg"))
D=list(food_path.glob("Egg/*.jpg"))          # *-shows all #(malignant/*)- shows all files inside the malignant folder
E=list(food_path.glob("Friedfood/*.jpg"))
F=list(food_path.glob("Meat/*.jpg"))
G=list(food_path.glob("Noodles-Pasta/*.jpg"))          # *-shows all #(malignant/*)- shows all files inside the malignant folder
H=list(food_path.glob("Rice/*.jpg"))
# I=list(food_path.glob("Seafood/*.jpg"))
# J=list(food_path.glob("Soup/*.jpg"))
# K=list(food_path.glob("Vegetable-Fruit/*.jpg"))
len(A),len(B),len(C),
len(D),len(E),len(F),
len(G),len(H),

food_dict={"Bread":A,
              "Dairyproduct":B,
              "Dessert":C,
              "Egg":D,
              "Friedfood":E,
              "Meat":F,
              "Noodles-Pasta":G,
              "Rice":H,}
food_class={"Bread":0,
              "Dairyproduct":1,
              "Dessert":2,
              "Egg":3,
              "Friedfood":4,
              "Meat":5,
              "Noodles-Pasta":6,
              "Rice":7,}


x=[]
y=[]

print("starting.....")
for i in food_dict:
  food_name=i
  food_path_list=food_dict[food_name]
  print("Image resizing....")
  for path in food_path_list:
    img=cv2.imread(str(path))
    img=cv2.resize(img,(224,224))
    img=img/255
    x.append(img)
    cls=food_class[food_name]
    y.append(cls)

len(x)
print("complete")
x=np.array(x)
y=np.array(y)

from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest=train_test_split(x,y,train_size=0.75,random_state=1)

len(xtrain),len(ytrain),len(xtest),len(ytest)

xtrain.shape

"""xtrain.shape,xtest.shape"""

xtrain.shape,xtest.shape

"""xtrain.shape,xtest.shape"""

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3),
                                               include_top=False,
                                               weights='imagenet')

print("[INFO] summary for base model...")
print(base_model.summary())

from tensorflow.keras.layers import MaxPooling2D
# from keras.layers.core import
from keras.layers import Dropout

from tensorflow.keras.models import Model
# construct the head of the model that will be placed on top of the
# the base model
headModel = base_model.output
headModel = MaxPooling2D(pool_size=(2, 2))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(32, activation="relu")(headModel)
headModel = Dropout(0.2)(headModel)
headModel = Dense(30, activation="softmax")(headModel)
# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=base_model.input, outputs=headModel)
# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
for layer in base_model.layers:
	layer.trainable = False

from tensorflow.keras.optimizers import Adam
# compile our model (this needs to be done after our setting our
# layers to being non-trainable)
print("[INFO] compiling model...")
opt = Adam(lr=1e-4)
model.compile(loss="sparse_categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])
# train the head of the network for a few epochs (all other layers
# are frozen) -- this will allow the new FC layers to start to become
# initialized with actual "learned" values versus pure random
print("[INFO] training head...")
#H = model.fit(
	#data_generator.flow(xtrain, ytrain, batch_size=32),
#	steps_per_epoch=len(xtrain) // 32,
	#validation_data=valAug.flow(xtest, ytest),
	#validation_steps=len(xtest) // 32,
#	epochs=5)

model_hist=model.fit(xtrain,ytrain,epochs=15,validation_data=(xtest,ytest),batch_size=180)

model.save("Model.h5")
# model.save("Model_keras_format")
# loaded_model = tf.keras.models.load_model("Model_keras_format")

# model.save('my_model.keras')
