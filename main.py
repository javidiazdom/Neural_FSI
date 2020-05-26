from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import RMSprop
from keras import backend as K
import keras
from time import time
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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
    "./Training",
    target_size=(288,162),
    batch_size= 400,
    class_mode='categorical'
)

x_batch, y_batch = next(train_generator)

plt.imshow(x_batch[0])
plt.show()

validation_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range= 10, 
    horizontal_flip=True
)
validation_generator = validation_datagen.flow_from_directory(
    "./Validation",
    target_size=(288,162),
    batch_size=300,
    class_mode='categorical'
)

model = Sequential()

model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(288,162,3)))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

epochs =  140

es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=10, restore_best_weights=True)

model.fit_generator(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator
    #callbacks=[es]
)

model.save("modelo.h5")