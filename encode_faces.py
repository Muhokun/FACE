from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os

###################################################TERMINAL###################################################
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True,
	help="Our Datasets")
ap.add_argument("-e", "--encodings", required=True,
	help="Our encoding")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
	help="Face model")
args = vars(ap.parse_args())

###################################################BOS BOS SEYLER###################################################
print("Reading faces !!!")
imagePaths = list(paths.list_images(args["dataset"]))


knownEncodings = []
knownNames = []
###################################################FOR Image###################################################

for (i, imagePath) in enumerate(imagePaths):

	print("Reading an image {}/{}".format(i + 1,
		len(imagePaths)))
	name = imagePath.split(os.path.sep)[-2]

	image = cv2.imread(imagePath)
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	
	boxes = face_recognition.face_locations(rgb,
		model=args["detection_method"])

	encodings = face_recognition.face_encodings(rgb, boxes)

################################################Encoding######################################################
	for encoding in encodings:
		
		knownEncodings.append(encoding)
		knownNames.append(name)
###################################################END###################################################
print("Encoding")
data = {"encodings": knownEncodings, "names": knownNames}
f = open(args["encodings"], "wb")
f.write(pickle.dumps(data))
f.close()
