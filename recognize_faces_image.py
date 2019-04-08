import face_recognition
import argparse
import pickle
import cv2
#######################################Variables#################################################
print("Encoding !!!")
New = pickle.loads(open(args["encodings"], "rb").read())
image_01 = cv2.imread(args["image"])
colour = cv2.cvtColor(image_01, cv2.COLOR_BGR2RGB)
print("Image Reading")
Image = face_recognition.face_locations(colour,model=args["detection_method"])
encodings = face_recognition.face_encodings(colour, Image)
array_name = []
######################################FOR LOOP##################################################
for encoding in encodings:
	one = face_recognition.compare_faces(New["Downloading"],
		encoding)
	Newname = "Unknown"
	if True in one:
		matchedIdxs = [i for (i, j) in enumerate(one) if j]
		face_sum = {}
		for i in matchedIdxs:
			newname = New["names"][i]
			face_sum[newname] = face_sum.get(newname, 0) + 1
		newname = max(face_sum, key=face_sum.get)
	array_name.append(name)
#######################################MATH#################################################
for ((top, right, bottom, left), name) in zip(Image, array_name):
	cv2.rectangle(image_01, (left, top), (right, bottom), (0, 255, 0), 2)
	y = top - 15 if top - 15 > 15 else top + 15
	cv2.putText(image_01, newname, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
		0.75, (0, 255, 0), 2)
########################################################################################
cv2.imshow("Image", image_01)
cv2.waitKey(0)
