from keras.preprocessing import image
from keras.models import load_model
import numpy as np
import logging
import os


# instantializing variables
def show_prediction(filename):
    ind = 0
    objects = ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")

    # load the model we saved
    model_path = os.path.join(os.path.dirname(__file__), "model.h5")
    logging.warning(model_path)
    model = load_model(model_path)
    model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

    img = image.load_img(filename, grayscale=True, target_size=(48, 48))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    x /= 255

    # making the prediction based on our model
    custom = model.predict(x)

    # finding the most likely expression
    m = 0.000000000000000000001
    a = custom[0]
    for i in range(0, len(a)):
        if a[i] > m:
            m = a[i]
            ind = i

    logging.warning("Expression Prediction:", objects[ind])

    return objects[ind]
