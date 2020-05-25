import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import keras
import os

labels =['Cubierto', 'Libro', 'Plato','Taza']

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

model = keras.models.load_model("modelo.h5")

pil_im = Image.open('./Validation/Tazas/IMG_20200524_130615_1.jpg','r')
im = np.asarray(pil_im.resize((162,288)))
plt.imshow(im)
print(im.shape)
plt.show()

im = im.reshape(1,288,162,3)
prediccion = model.predict(im)

y_classes = prediccion.argmax(axis=-1)
labels.sort()

print(labels[y_classes[0]])