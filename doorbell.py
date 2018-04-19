# USAGE
# python doorbell.py --conf conf.json

# import the necessary packages
from doorbell_lib.tempimage import TempImage
from dropbox.client import DropboxOAuth2FlowNoRedirect
from dropbox.client import DropboxClient
from picamera.array import PiRGBArray
from picamera import PiCamera
import numpy as np
import argparse
import warnings
import datetime
import imutils
import json
import time
import cv2
import sys, os
import boto3
from omxplayer import OMXPlayer
#import cognitive_face as azure_cf


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True,
	help="path to the JSON configuration file")
args = vars(ap.parse_args())

# filter warnings, load the configuration and initialize the Dropbox
# client
warnings.filterwarnings("ignore")
conf = json.load(open(args["conf"]))
client = None

rekognition = boto3.client('rekognition');

azure_key = 'b5abb2e46fa142e3b2197e70f242f7cb'
azure_url = 'https://westcentralus.api.cognitive.microsoft.com/face/v1.0/detect?returnFaceId=true&returnFaceLandmarks=true'
#azure_cf.Key.set(azure_key)
#azure_cf.BaseUrl.set(azure_url)

# CNN Model Constructs
# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
        "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

PROTOTXT = 'MobileNetSSD_deploy.prototxt.txt'
MODEL = 'MobileNetSSD_deploy.caffemodel'

CONFIDENCE = 0.2


# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)



# path to cascade xml configuration file
#cascPath = "/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml"
#cascPath = "/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml"

file_path_or_url = conf["doorbellSound"]

# check to see if the Dropbox should be used
if conf["use_dropbox"]:
	# connect to dropbox and start the session authorization process
	flow = DropboxOAuth2FlowNoRedirect(conf["dropbox_key"], conf["dropbox_secret"])
	print "[INFO] Authorize this application: {}".format(flow.start())
	authCode = raw_input("Enter auth code here: ").strip()

	# finish the authorization and grab the Dropbox client
	(accessToken, userID) = flow.finish(authCode)
	client = DropboxClient(accessToken)
	print "[SUCCESS] dropbox account linked"

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = tuple(conf["resolution"])
camera.framerate = conf["fps"]
rawCapture = PiRGBArray(camera, size=tuple(conf["resolution"]))

# allow the camera to warmup, then initialize the average frame, last
# uploaded timestamp, and frame motion counter
print "[INFO] warming up..."
time.sleep(conf["camera_warmup_time"])
avg = None
lastUploaded = datetime.datetime.now()
motionCounter = 0

# capture frames from the camera
for f in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	# grab the raw NumPy array representing the image and initialize
	# the timestamp and occupied/unoccupied text
	frame = f.array

	timestamp = datetime.datetime.now()
	text = "No Visitors"
	motionFound = "NO"

	# resize the frame, convert it to grayscale, and blur it
	frame = imutils.resize(frame, width=500)

	if conf["motion_enabled"]:
		frameHeight, frameWidth = frame.shape[:2]
	
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		gray = cv2.GaussianBlur(gray, (21, 21), 0)
	
		# if the average frame is None, initialize it
		if avg is None:
			print "[INFO] starting background model..."
			avg = gray.copy().astype("float")
			rawCapture.truncate(0)
			continue
	
		# accumulate the weighted average between the current frame and
		# previous frames, then compute the difference between the current
		# frame and running average
		cv2.accumulateWeighted(gray, avg, 0.5)
		frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))
	
		# threshold the delta image, dilate the thresholded image to fill
		# in holes, then find contours on thresholded image
		thresh = cv2.threshold(frameDelta, conf["delta_thresh"], 255, cv2.THRESH_BINARY)[1]
		#th3 = cv2.adaptiveThreshold(frameDelta,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
		thresh = cv2.dilate(thresh, None, iterations=2)
		(image, cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	
		# loop over the contours
		for c in cnts:
			# if the contour is too small, ignore it
			if cv2.contourArea(c) < conf["min_area"]:
				continue
	
			# compute the bounding box for the contour, draw it on the frame,
			# and update the text
			(x, y, w, h) = cv2.boundingRect(c)
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
			motionHeight = x + h
        		print "Frame Width, Frame Height: " + str(frameWidth) + "," + str(frameHeight)
			print "X, Y, W, H: " + str(x) + "," + str(y) + "," + str(w) + "," + str(h)
			text =  "X, Y, W, H: " + str(x) + "," + str(y) + "," + str(w) + "," + str(h)
			print "Motion height: " + str(y + h)
			print "Motion width: " + str(x + w)
			if y < conf["motionYThreshold"]:
				print "Motion in the street ... ignoring"
				text =  "Motion in Street - X, Y, W, H: " + str(x) + "," + str(y) + "," + str(w) + "," + str(h)
				continue
			else:
				print "Motion within yard, alerting"
				text =  "Motion in Yard - X, Y, W, H: " + str(x) + "," + str(y) + "," + str(w) + "," + str(h)
	
			#text = "Visitors Found"
			motionFound = "YES"

		# draw the text and timestamp on the frame
		ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
		cv2.putText(frame, "Frontdoor Status: {}".format(text), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
		cv2.putText(frame, ts, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)


        # Determine if a person is there
        # grab the frame dimensions and convert it to a blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	detections = net.forward()

	# loop over the detections
	for i in np.arange(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
		if confidence > CONFIDENCE:
			# extract the index of the class label from the
			# `detections`, then compute the (x, y)-coordinates of
			# the bounding box for the object
			idx = int(detections[0, 0, i, 1])
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# draw the prediction on the frame
			label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
			cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

			if (timestamp - lastUploaded).seconds >= conf["min_upload_seconds"]:
				# write the image to temporary file
				t = TempImage()
				cv2.imwrite(t.path, frame)

				try:
					with open(t.path, 'rb') as image:
						os.system('echo "Ding Dong: " | mailx -s "Person Detected!" -A ' + t.path + ' jbf2000ru@gmail.com')
				except Exception, e:
					os.system('echo ' + str(e) + ' | mailx -s "Person Detection Doorbell Exception Caught" -A ' + t.path + ' jbf2000ru@gmail.com')
					print "Exception caught calling AWS: " + str(e)


			# Highlight the face
			faceCascade = cv2.CascadeClassifier(conf["cascPath"])
			faces = faceCascade.detectMultiScale(
				frame,
				scaleFactor=conf["msScaleFactor"],
				minNeighbors=conf["msMinNeighbors"],
				minSize=(conf["msMinSize"], conf["msMinSize"]),
				flags = cv2.CASCADE_SCALE_IMAGE
			)
                   
			print "Found {0} faces!".format(len(faces))
                   				 
			# Draw a rectangle around the faces
			for (x, y, w, h) in faces:
				cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

			if (len(faces) > 0):
				text = "Visitor here!"
				# draw the text and timestamp on the frame
				ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
				cv2.putText(frame, "Frontdoor Status: {}".format(text), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
				cv2.putText(frame, ts, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

				if (timestamp - lastUploaded).seconds >= conf["min_upload_seconds"]:
					# write the image to temporary file
					t = TempImage()
					cv2.imwrite(t.path, frame)
		
					#print "Inside TS code: Found {0} faces!".format(len(faces))
					#os.system('mpack -s "Face Found!!!" ' + t.path + ' jbf2000ru@gmail.com')
					#os.system('echo "Found a face" | mailx -s "FACE FOUND!!!" -A ' + t.path + ' jbf2000ru@gmail.com lindelfeldman@gmail.com')
			
					try:
						with open(t.path, 'rb') as image:
							response = rekognition.detect_faces(Image={'Bytes': image.read()}, Attributes=['ALL'])
							print response
							os.system('echo ' + str(response) + ' | mailx -s "FACE FOUND (AWS RESPONSE)!!!!" -A ' + t.path + ' jbf2000ru@gmail.com lindelfeldman@gmail.com')
					except Exception, e:
						os.system('echo ' + str(e) + ' | mailx -s "AWS Doorbell Exception Caught" -A ' + t.path + ' jbf2000ru@gmail.com')
						print "Exception caught calling AWS: " + str(e)
		
					#try:
						#a_response = azure_cf.face.detect(t.path)
						#if a_response is None:
						#	print 'Response is null?'
						#else: 
						#	print a_response
						#	os.system('echo ' + str(a_response) + ' | mailx -s "FACE FOUND (AZURE RESPONSE)!!!!" -A ' + t.path + ' jbf2000ru@gmail.com')
		
		
					#except Exception, e:
						#os.system('echo ' + str(e) + ' | mailx -s "Azure Doorbell Exception Caught" -A ' + t.path + ' jbf2000ru@gmail.com')
						#print "Exception caught calling AZURE: " + str(e)
		

	# check to see if the room is occupied
	if conf["motion_enabled"]:
		if motionFound == "YES":
			# check to see if enough time has passed between uploads
			if (timestamp - lastUploaded).seconds >= conf["min_upload_seconds"]:
				# write the image to temporary file
				t = TempImage()
				cv2.imwrite(t.path, frame)
	
				# increment the motion counter
				motionCounter += 1
	
				# check to see if the number of frames with consistent motion is
				# high enough
				if motionCounter >= conf["min_motion_frames"]:
	
					# check to see if dropbox sohuld be used
					if conf["use_dropbox"]:
						# upload the image to Dropbox and cleanup the tempory image
						print "[UPLOAD] {}".format(ts)
						path = "{base_path}/{timestamp}.jpg".format(
							base_path=conf["dropbox_base_path"], timestamp=ts)
						client.put_file(path, open(t.path, "rb"))
						#t.cleanup()
					else:
						print ("Sending a moo")
						#os.system('omxplayer -o local --vol 900 /home/pi/wavs/cow.wav')
						player = OMXPlayer(file_path_or_url)
						player.play()
	
						print ("sending ding dong email ")
						try:
    							with open(t.path, 'rb') as image:
        							#response = rekognition.detect_labels(Image={'Bytes': image.read()})
								#os.system('echo ' + str(response) + ' | mailx -s "Ding Dong: Someone is here" -A ' + t.path + ' jbf2000ru@gmail.com lindelfeldman@gmail.com')
								# CHeck AWS Facial detection just in case local algorithm missed
        							#response = rekognition.detect_faces(Image={'Bytes': image.read()}, Attributes=['ALL'])
        							#print response
								#os.system('echo ' + str(response) + ' | mailx -s "Ding Dong: Someone is here" -A ' + t.path + ' jbf2000ru@gmail.com lindelfeldman@gmail.com')
								os.system('echo "Ding Dong: " | mailx -s "Ding Dong: Someone is here" -A ' + t.path + ' jbf2000ru@gmail.com')
		
						except Exception, e:
							os.system('echo ' + str(e) + ' | mailx -s "Doorbell Exception Caught" -A ' + t.path + ' jbf2000ru@gmail.com')
							print "Exception caught calling AWS: " + str(e)
	
					# update the last uploaded timestamp and reset the motion
					# counter
					lastUploaded = timestamp
					motionCounter = 0
				else: 
					t.cleanup()
	
		# otherwise, the room is not occupied
		else:
			motionCounter = 0
	
	# check to see if the frames should be displayed to screen
	if conf["show_video"]:
		# display the security feed
		cv2.imshow("Security Feed", frame)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key is pressed, break from the lop
		if key == ord("q"):
			break

	# clear the stream in preparation for the next frame
	rawCapture.truncate(0)
