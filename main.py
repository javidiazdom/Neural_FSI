from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import RMSprop
from keras import backend as K
import keras
from time import time

import matplotlib.pyplot as plt

train_datagen = ImageDataGenerator(
    rescale=1./255, 
    rotation_range= 10, 
    width_shift_range=0.1, 
    height_shift_range=0.1, 
    zoom_range= 0.1,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    "./Dataset",
    target_size=(288,162),
    batch_size= 70,
    class_mode='categorical'
)

x_batch, y_batch = next(train_generator)

for i in range(0,19):
    image = x_batch[i]
    plt.imshow(image)
    plt.show()