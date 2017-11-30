import cv2
import numpy as np
import math
from PIL import Image
import os
import time
import sys

cam = cv2.VideoCapture(0)
backfilter = cv2.BackgroundSubtractorMOG()

prev_pos = []
count_vec = 0
prev_num = []
prev_dir = []

# points for skin colour detection (these co-ordinates are for 720X1280 video feed)
points_x = [250,250,250,250,250,250,250,250,250,250,210,210,210,210,210,210,210,210]
points_y = [250,200,220,230,180,160,140,120,100,80,230,220,210,200,190,180,160,140]

for i in range(0,10):
	for j in range(0,10):
		points_x.append(200+10*i)
		points_y.append(300+10*j)

# initialisation for high and low values for background subtraction (the values are for hsv image)
high = [0*255,0*255,0*255]
low = [1*255,1*255,1*255]

points_x = np.array(points_x).astype('int')
points_y = np.array(points_y).astype('int')

ret, frame = cam.read()

# conversion of points for general video dimensions
dimension_x = frame.shape[0]
dimension_y = frame.shape[1]
points_x = points_x*(float(dimension_x)/720.0)
points_y = points_y*(float(dimension_y)/1280.0)

points_x = points_x.astype('int')
points_y = points_y.astype('int')


count_defects_t = 0 
while (1):
	ret, frame = cam.read()

	# displaying points for skin colour detection
	for i in range(len(points_x)):
		cv2.circle(frame,tuple((points_x[i],points_y[i])),1,[255,255,255],3)
	cv2.imshow('',cv2.flip(frame,1))
	hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	hsv_array = np.array(list(hsv_image)).astype('uint8')
	k = cv2.waitKey(30) &0xff
	if k==27:

		# storing the range for skin colour of the user
		for i in range(len(points_x)):
			for j in range(3):
				if (high[j] < hsv_array[points_x[i],points_y[i],j]):
					high[j] = hsv_array[points_x[i],points_y[i],j]
				if (low[j] > hsv_array[points_x[i],points_y[i],j]):
					low[j] = hsv_array[points_x[i],points_y[i],j]
		break

average_size = 2	# setting the size for 2D kernel to be used later
high = np.array(high).astype('int')
low = np.array(low).astype('int')

while (1):
	ret, frame = cam.read()
	back_image = backfilter.apply(frame)
	hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	result_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	num_row = 0
	hsv_array = np.array(list(hsv_image))
	result_array = np.array(list(result_image))
	blur = cv2.GaussianBlur(hsv_array,(35,35),0)		# applying gaussian blur

	# background subtraction
	con1 = abs(hsv_array[:,:,0]-((low[0])+high[0])/2)<((high[0])-low[0])/2
	con2 = abs(hsv_array[:,:,1]-((low[1])+high[1])/2)<((high[1])-low[1])/2
	con3 = abs(hsv_array[:,:,2]-((low[2])+high[2])/2)<((high[2])-low[2])/2
	con4 = np.logical_and(con1,con2,con3)
	result_array = np.where(con4,255,0)
	result_array = result_array.astype('uint8')

	blur = cv2.GaussianBlur(result_array,(35,35),0)		# applying gaussian blur
	
	kernel = np.ones((average_size,average_size),np.float32)/((average_size**2)/4)		
	dst = cv2.filter2D(blur,-1,kernel)		# applying a 2D filter on the image

	binary = np.where(dst>0,255,0).astype('uint8')	# converting the image to a binary image
	binary3 = np.copy(binary)
	
	# finding contours for a window within the image
	dimensions = np.array([int(400*(float(dimension_x)/720.0)),int(640*(float(dimension_y)/1280.0))])
	binary3 = binary3[:dimensions[0],:dimensions[1]]		
	contours, hierarchy = cv2.findContours(binary3,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	binary3 = cv2.drawContours(binary,contours,-1,(0,255,0),3)


	image = np.zeros((720,1280,3))
	image[:,:,0] = binary[:,:]

	if (len(contours) > 0):

		#find the max counter in the window
		c = max(contours, key=lambda x:cv2.contourArea(x))
		cv2.drawContours(image, [c], -1, (0,255,0), 0)

		# find the convex hull in the image
		hull = cv2.convexHull(c)
		hull2 = cv2.convexHull(c,returnPoints=False)
		
		bool = False
		point_window = dimensions/10

		# computing the direction of motion of the hand (vertical or horizontal)
		current_position = []
		for x in hull:

			if bool:
				b = True
				if (current_position != []):
					for y in current_position:

						if (abs(y[0][0]-x[0][0])<=point_window[0]) and (abs(y[0][1]-x[0][1])<=point_window[1]):
							b = False
							break
						if (abs(dimensions[0]-x[0][0])<=point_window[0]) or 	(abs(dimensions[1]-x[0][1])<=point_window[1]):
							b = False
							break
					if b:
						current_position.append(x)
			else:
				bool = True
				current_position.append(x)

		vec = []
		hor_sum = 0
		ver_sum = 0

		if (prev_pos != []):
			for i in range(min(len(prev_pos),len(current_position))):
				cv2.line(image,(prev_pos[i][0][0],prev_pos[i][0][1]),(current_position[i][0][0],current_position[i][0][1]),(255,255,255),3)

				hor_sum += abs(current_position[i][0][0] - prev_pos[i][0][0])
				ver_sum += abs(current_position[i][0][1] - prev_pos[i][0][1])

		prev_pos = current_position

		cnt_vec = 0

		dir_ans = 0


		# conditions for detecting horizontal motion
		if(ver_sum > 2*hor_sum):
			dir_ans = 1

		elif(hor_sum > 2*ver_sum):
			dir_ans = 2
		
		if(len(prev_dir) < 2):
			prev_dir.append(dir_ans)
		else:
			prev_dir.pop(0)
			prev_dir.append(dir_ans)
		if(len(prev_dir) == 2):
			temp = prev_dir[0]
			bo = True
			for x in range(1,2):
				if(prev_dir[x] != temp):
					bo = False
					break
			if(bo):
				if(temp == 1):
					cv2.putText(image,"Vertical", (500, 100), cv2.FONT_HERSHEY_SIMPLEX,2,color=(255,255,255))
				elif(temp == 2):
					cv2.putText(image,"horizontal", (500, 100), cv2.FONT_HERSHEY_SIMPLEX,2,color=(255,255,255))

				else:
					cv2.putText(image,"Random", (500, 100), cv2.FONT_HERSHEY_SIMPLEX,2,color=(255,255,255))

		# drawing the countours in the image
		cv2.drawContours(image, [hull], -1, (0,0,255), 0)
		cv2.drawContours(image, hull, -1, (255,255,255), 3)

		if len(hull)<3 or len(c)<3:
			continue

		defects = cv2.convexityDefects(c,hull2)		# points except the ones on the convex hull in the contour

		count_defects = 0
		try:
			for i in range(defects.shape[0]):
				s,e,f,d = defects[i,0]

				# removing angles larger than 90 deg (to remove defects that are not needed)
				start = tuple(c[s][0])
				end = tuple(c[e][0])
				far = tuple(c[f][0])


				a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
				b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
				co = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)


				angle = math.acos((b**2 + co**2 - a**2)/(2*b*co)) * 57


				if angle <= 90:
				    count_defects += 1
				    cv2.circle(image, far, 1, [0,0,255], -1)

				cv2.line(image,start, end, [0,255,0], 2)
			if(len(prev_num) <5):
				prev_num.append(count_defects)
			else:
				prev_num.pop(0)
				prev_num.append(count_defects)
			num_defects = 0

			if(len(prev_num) == 5):
				temp = prev_num[0]
				bo = True
				for x in range(1,5):
					if(prev_num[x] != temp):
						bo = False
						break
				if(bo):
					num_defects = temp

			# conditions for different gestures
			if num_defects == 1:
			    cv2.putText(image,"I am Two", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
			elif num_defects == 2:
				str = "I am Three"
				cv2.putText(image, str, (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
			elif num_defects == 3:
			    cv2.putText(image,"I am Four", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
			elif num_defects == 4:
			    cv2.putText(image,"I am Five", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
			else:
			    cv2.putText(image,"No Gesture Detected!!", (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 2, 2)

			count_defects_t = count_defects

		except:
			continue

	cv2.imshow('',image)
	k = cv2.waitKey(30) & 0xff
	if k==27:
		break

cam.release()
cv2.destroyAllWindows()