# USAGE
# python classify.py --model pokedex.model --labelbin lb.pickle --image examples/charmander_counter.png

# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras import backend as K
import numpy as np
import argparse
import pickle
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

# load the image
image = cv2.imread(args["image"])
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
output = image.copy()
 
# pre-process the image for classification
image = cv2.resize(image, (80,60))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

# load the trained convolutional neural network and the label binarizer
print("[INFO] loading network...")
model = load_model("pokedex_test.model")
lb = pickle.loads(open("lb_test.pickle", "rb").read())

# classify the input imag
proba = model.predict(image)[0]
idx = np.argmax(proba)
label = lb.classes_[idx]

# we'll mark our prediction as "correct" of the input image filename
# contains the predicted label text (obviously this makes the
# assumption that you have named your testing image files this way)
#filename = args["image"][args["image"].rfind(os.path.sep) + 1:]
#correct = "correct" if filename.rfind(label) != -1 else "incorrect"

# build the label and draw the label on the image
label = "{}".format(label)
#output = imutils.resize(output, width=400)
#cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
#	0.7, (0, 255, 0), 2)

# show the output image
print("Output is : {}".format(label))
#cv2.imshow("Output", output)
#cv2.waitKey(0)

K.clear_session()