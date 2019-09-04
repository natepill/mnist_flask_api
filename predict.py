import tensorflow as tf
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from PIL import Image


model = load_model('milad_model.h5')
# The default Graph being used in the current thread.
graph = tf.get_default_graph()

img = Image.open('milad.png')
# convert to array
img = img_to_array(img.resize((28, 28)))
# reshape args(num of img, size, size,
x = img.reshape(1, 28, 28, 1)
x /= 255

with graph.as_default():
    out = model.predict(x)


print(out[0])
r = np.argmax(out[0])
print(r)
