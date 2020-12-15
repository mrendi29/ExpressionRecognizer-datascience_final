from keras.preprocessing import image
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import logging

# instantializing variables
def show_prediction(filename):
    ind = 0
    objects = ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")

    # load the model we saved
    model = load_model("model.h5")
    model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

    img = image.load_img(filename, grayscale=True, target_size=(48, 48))
    show_img = image.load_img(filename, grayscale=False, target_size=(200, 200))
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

    plt.title(objects[ind])
    plt.gray()
    plt.imshow(show_img)
    plt.axis("off")
    plt.show()

    return objects[ind]
