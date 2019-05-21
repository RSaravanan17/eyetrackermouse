from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import pyautogui
#import tkinter as tk

def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
 
	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])
 
	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)
 
	# return the eye aspect ratio
	return ear



# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="",
	help="path to input video file")
args = vars(ap.parse_args())


# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold
 
# initialize the frame counters and the total number of blinks
COUNTER = 0
TOTAL = 0



def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])

	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)

	# return the eye aspect ratio
	return ear
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="",
	help="path to input video file")
args = vars(ap.parse_args())
 
# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold
EYE_AR_THRESH = 0.32
EYE_AR_CONSEC_FRAMES = 2

# initialize the frame counters and the total number of blinks
COUNTER = 0
TOTAL = 0

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])


#(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
#(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

(lStart, lEnd) = (18, 23)
(rStart, rEnd) = (23, 28)
# start the video stream thread
print("[INFO] starting video stream thread...")
#vs = FileVideoStream(args["video"]).start()
fileStream = True
vs = VideoStream(src=1).start()


# vs = VideoStream(usePiCamera=True).start()
fileStream = False
time.sleep(1.0)

#vs = cv2.VideoCapture(0)
#calibrate over here
print ("look down comfortably and press enter when ready")
left = raw_input()
frame = vs.read()
rows, cols, _ = frame.shape
frame = imutils.resize(frame, width=900)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
gray_blur = cv2.GaussianBlur(gray, (7, 7), 0)
	
'''_, threshold = cv2.threshold(gray_blur, 30, 255, cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

for cnt in contours:
	(x, y, w, h) = cv2.boundingRect(cnt)
	#cv2.drawContours(roi, [cnt], -1, (0, 0, 255), 3)
	cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
	cv2.line(frame, (x + int(w/2), 0), (x + int(w/2), rows), (0, 255, 0), 2)
	cv2.line(frame, (0, y + int(h/2)), (cols, y + int(h/2)), (0, 255, 0), 2)
break'''


#cv2.imshow("gray roi", gray_blur)

# detect faces in the grayscale frame
rects = detector(gray, 0)
# loop over the face detections
for rect in rects:
	# determine the facial landmarks for the face region, then
	# convert the facial landmark (x, y)-coordinates to a NumPy
	# array
	shape = predictor(gray, rect)
	shape = face_utils.shape_to_np(shape)

	# extract the left and right eye coordinates, then use the
	# coordinates to compute the eye aspect ratio for both eyes
	leftEye = shape[lStart:lEnd]
	rightEye = shape[rStart:rEnd]

	minX = 9999999
	maxX = -9999999
	minY = 9999999
	maxY = -9999999
	for point in rightEye:
		minX = min(minX, point[0])
		maxX = max(maxX, point[0])
		minY = min(minY, point[1])
		maxY = max(maxY, point[1])
	print (maxX - minX)
	print (maxY - minY)
	print ("-----")
	leftEyeRoi = gray_blur[minY:maxY, minX:maxX]
	_, threshold = cv2.threshold(leftEyeRoi, 30, 255, cv2.THRESH_BINARY_INV)
 	contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
 	contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

 	xmid = None
 	ymid = None
 	for cnt in contours:
 		(x, y, w, h) = cv2.boundingRect(cnt)
 		xmid = float(x) + float(w)/2
 		ymid = float(y) + float(h)/2

 		#cv2.drawContours(roi, [cnt], -1, (0, 0, 255), 3)
 		#cv2.rectangle(threshold, (x, y), (x + w, y + h), (255, 0, 0), 2)
 		cv2.line(threshold, (x + int(w/2), 0), (x + int(w/2), rows), (0, 255, 0), 2)
 		cv2.line(threshold, (0, y + int(h/2)), (cols, y + int(h/2)), (0, 255, 0), 2)
		break

calibratedBottom = maxY + minY

print ("calibratedBottom: " + str(calibratedBottom))

print ("look up comfortably and press enter when ready: ")
right = raw_input()
frame = vs.read()
rows, cols, _ = frame.shape
frame = imutils.resize(frame, width=900)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
gray_blur = cv2.GaussianBlur(gray, (7, 7), 0)
	
'''_, threshold = cv2.threshold(gray_blur, 30, 255, cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

for cnt in contours:
	(x, y, w, h) = cv2.boundingRect(cnt)
	#cv2.drawContours(roi, [cnt], -1, (0, 0, 255), 3)
	cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
	cv2.line(frame, (x + int(w/2), 0), (x + int(w/2), rows), (0, 255, 0), 2)
	cv2.line(frame, (0, y + int(h/2)), (cols, y + int(h/2)), (0, 255, 0), 2)
break'''


#cv2.imshow("gray roi", gray_blur)

# detect faces in the grayscale frame
rects = detector(gray, 0)
# loop over the face detections
for rect in rects:
	# determine the facial landmarks for the face region, then
	# convert the facial landmark (x, y)-coordinates to a NumPy
	# array
	shape = predictor(gray, rect)
	shape = face_utils.shape_to_np(shape)

	# extract the left and right eye coordinates, then use the
	# coordinates to compute the eye aspect ratio for both eyes
	leftEye = shape[lStart:lEnd]
	rightEye = shape[rStart:rEnd]

	minX = 9999999
	maxX = -9999999
	minY = 9999999
	maxY = -9999999
	for point in rightEye:
		minX = min(minX, point[0])
		maxX = max(maxX, point[0])
		minY = min(minY, point[1])
		maxY = max(maxY, point[1])

	print (maxX - minX)
	print (maxY - minY)
	print ("-----")
	leftEyeRoi = gray_blur[minY:maxY, minX:maxX]
	_, threshold = cv2.threshold(leftEyeRoi, 30, 255, cv2.THRESH_BINARY_INV)
 	contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
 	contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

 	xmid = None
 	ymid = None
 	for cnt in contours:
 		(x, y, w, h) = cv2.boundingRect(cnt)
 		xmid = float(x) + float(w)/2
 		ymid = float(y) + float(h)/2

 		#cv2.drawContours(roi, [cnt], -1, (0, 0, 255), 3)
 		#cv2.rectangle(threshold, (x, y), (x + w, y + h), (255, 0, 0), 2)
 		cv2.line(threshold, (x + int(w/2), 0), (x + int(w/2), rows), (0, 255, 0), 2)
 		cv2.line(threshold, (0, y + int(h/2)), (cols, y + int(h/2)), (0, 255, 0), 2)
		break
calibratedTop = maxY + minY
print ("calibratedTop: " + str(calibratedTop))

#root = tk.Tk()

screen_width = 2560
screen_height = 1600



# loop over frames from the video stream
while True:
	#ret, frame2 = cap.read()
	#if ret is False:
	#	break
	#roi = frame2[269: 795, 537: 1416]
	#rows, cols, _ = roi.shape
	#gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)






	# if this is a file video stream, then we need to check if
	# there any more frames left in the buffer to process
	if fileStream and not vs.more():
		break
 
	# grab the frame from the threaded video file stream, resize
	# it, and convert it to grayscale
	# channels)
	frame = vs.read()
	rows, cols, _ = frame.shape
	frame = imutils.resize(frame, width=900)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray_blur = cv2.GaussianBlur(gray, (7, 7), 0)
 	
 	'''_, threshold = cv2.threshold(gray_blur, 30, 255, cv2.THRESH_BINARY_INV)
 	contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
 	contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

 	for cnt in contours:
 		(x, y, w, h) = cv2.boundingRect(cnt)
 		#cv2.drawContours(roi, [cnt], -1, (0, 0, 255), 3)
 		cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
 		cv2.line(frame, (x + int(w/2), 0), (x + int(w/2), rows), (0, 255, 0), 2)
 		cv2.line(frame, (0, y + int(h/2)), (cols, y + int(h/2)), (0, 255, 0), 2)
		break'''

	
	#cv2.imshow("gray roi", gray_blur)

	# detect faces in the grayscale frame
	rects = detector(gray, 0)
	# loop over the face detections
	for rect in rects:
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
 
		# extract the left and right eye coordinates, then use the
		# coordinates to compute the eye aspect ratio for both eyes
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]

		minX = 9999999
		maxX = -9999999
		minY = 9999999
		maxY = -9999999
		for point in rightEye:
			minX = min(minX, point[0])
			maxX = max(maxX, point[0])
			minY = min(minY, point[1])
			maxY = max(maxY, point[1])
		#print (maxX - minX)
		#print (maxY - minY)
		#print ("-----")
		leftEyeRoi = gray_blur[minY:maxY, minX:maxX]
		_, threshold = cv2.threshold(leftEyeRoi, 30, 255, cv2.THRESH_BINARY_INV)
	 	contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	 	contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

	 	xmid = None
	 	ymid = None
	 	for cnt in contours:
	 		(x, y, w, h) = cv2.boundingRect(cnt)
	 		xmid = float(x) + float(w)/2
	 		ymid = float(y) + float(h)/2

	 		#cv2.drawContours(roi, [cnt], -1, (0, 0, 255), 3)
	 		#cv2.rectangle(threshold, (x, y), (x + w, y + h), (255, 0, 0), 2)
	 		cv2.line(threshold, (x + int(w/2), 0), (x + int(w/2), rows), (0, 255, 0), 2)
	 		cv2.line(threshold, (0, y + int(h/2)), (cols, y + int(h/2)), (0, 255, 0), 2)
			break
		if threshold is not None and xmid is not None:
			print (threshold.shape[0])
			print (threshold.shape[1])

			#curPos = pyautogui.position()
			ymid = maxY + minY
			print ("max + min: " + str(ymid))

			val = float(screen_height) * ((ymid - float(calibratedTop))/(float(calibratedBottom) - float(calibratedTop)))
			print ("val: " + str(val))
			pyautogui.moveTo(None, val)
			#cv2.imshow("Threshold", threshold)
			#cv2.moveWindow("Threshold", 0, 0)



		


		#leftEAR = eye_aspect_ratio(leftEye)
		#rightEAR = eye_aspect_ratio(rightEye)
 
		# average the eye aspect ratio together for both eyes
		#ear = (leftEAR + rightEAR) / 2.0


		# compute the convex hull for the left and right eye, then
		# visualize each of the eyes
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

		# check to see if the eye aspect ratio is below the blink
		# threshold, and if so, increment the blink frame counter
		'''if ear < EYE_AR_THRESH:
			COUNTER += 1
 
		# otherwise, the eye aspect ratio is not below the blink
		# threshold
		else:
			# if the eyes were closed for a sufficient number of
			# then increment the total number of blinks
			if COUNTER >= EYE_AR_CONSEC_FRAMES:
				#pyautogui.click()
				TOTAL += 1
 
			# reset the eye frame counter
			COUNTER = 0'''

		# draw the total number of blinks on the frame along with
		# the computed eye aspect ratio for the frame
		cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		#cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
		#	cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
 		break


	# show the frame
	cv2.imshow("Frame", frame)
	cv2.moveWindow("Frame", 400, 0)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
 
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()