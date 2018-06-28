from keras.models import model_from_json
from image_read import load_images
import numpy as np
import cv2
from find_hands import findObjects, drawRectHands

# for testing a set of images
#image_testing = "Test189x110/27734/*.jpg"          # set of pictures
#image_testing = "Test189x110/27734/7350944.jpg"    # normal picture
image_testing = "Test189x110/27734/73515944.jpg"   # fliped picture

test_set = np.array(load_images(image_testing, blackAndWhite=False))

# loading json and create model
with open('CNN_model.json', 'r') as json_file:
    loaded_model_json = json_file.read()
model = model_from_json(loaded_model_json)
# loading weights into new model
model.load_weights("CNN_model.h5")
print("\nLoaded model from disk\n")

model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])

# for every picture in a folder find hands:
for img in test_set:
    hands = findObjects(model, img)
    drawRectHands(img, hands)
    cv2.imshow("Output", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()