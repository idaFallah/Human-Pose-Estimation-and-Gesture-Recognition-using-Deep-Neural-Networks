# MPII focuses on human activities, MPII is based on VGG_net

# importing libs

import cv2
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow
import numpy as np

from google.colab import drive
drive.mount('/content/drive')

image = cv2.imread('/content/drive/MyDrive/Computer Vision/Images/megan.jpg')

cv2_imshow(image)

image.shape, image.shape[0] * image.shape[1] * 3

type(image)

# reversing the image's dimension to be able to send  it to our NN model

image_blob = cv2.dnn.blobFromImage(image=image, scalefactor=1.0/255, size=(image.shape[1], image.shape[0]))

type(image_blob), image_blob.shape  # image blob is in batch format, 1 in (1, 3, 337, 600) indicates that we send 1 image at a time

# loading the pretrained NN(caffe DL framework)

network = cv2.dnn.readNetFromCaffe('/content/drive/MyDrive/Computer Vision/Weights/pose_deploy_linevec_faster_4_stages.prototxt',
                                   '/content/drive/MyDrive/Computer Vision/Weights/pose_iter_160000.caffemodel')    # first param is the path to NN structure, second is path to weights

network.getLayerNames()

len(network.getLayerNames())

network.setInput(image_blob)
output = network.forward()  # the imageis sent thru input layer & will go thru the layers till it reaches the vector layers in the end, we won't have 1000 classes thu cuz we're not classifying the images, we will have 15 classes to identify the 15 points in human body

output.shape # the first value = number of images in the batch, the second value is info abut the points that were detected, the last 2 vlaues mean where the points are in the image

position_width = output.shape[3]
position_height = output.shape[2]

position_width, position_height

num_points = 15
points = []
threshold = 0.1
for i in range(num_points):
  confidence_map = output[0, i, :, :]  # possibilities for each of the 15 important points in body
  #print(confidence_map)
  #print(len(confidence_map))
  _, confidence, _, point = cv2.minMaxLoc(confidence_map) # minmaxloc returns min point and max point then their positions, we want the max value not min
  #print(confidence)
  #print(point)

  # the location of points are so far from the size of the image so we need to do some transformation on them
  x = int((image.shape[1] * point[0]) / position_width)
  y = int((image.shape[0] * point[1]) / position_height)
  #print(x, y)
  if confidence > threshold:
    cv2.circle(image, (x, y), 5, (0, 255, 0), thickness=-1)
    cv2.putText(image, '{}'.format(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255))
    points.append((x, y))
  else:
    points.append(None)

points

plt.figure(figsize=(14, 10))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB));

# connecting the points

point_connections = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 14],
                     [14, 11], [14, 8], [8, 9], [11, 12], [9, 10], [12, 13]]

point_connections

for connection in point_connections:
  #print(connection)
  partA = connection[0]
  partB = connection[1]
  #print(partA, partB)
  if points[partA] and points[partB]:
    cv2.line(image, points[partA], points[partB], (255, 0, 0))

plt.figure(figsize=(14, 10))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB));

# detecting movements(if the arms are above the head)

image2 = cv2.imread('/content/drive/MyDrive/Computer Vision/Images/player.jpg')
cv2_imshow(image2)

image_blob2 = cv2.dnn.blobFromImage(image = image2, scalefactor = 1.0 / 255, size = (image2.shape[1], image2.shape[0]))
network.setInput(image_blob2)
output2 = network.forward()
position_width = output2.shape[3]
position_height = output2.shape[2]
num_points = 15
points = []
threshold = 0.1
for i in range(num_points):
  confidence_map = output2[0, i, :, :]
  _, confidence, _, point = cv2.minMaxLoc(confidence_map)
  x = int((image2.shape[1] * point[0]) / position_width)
  y = int((image2.shape[0] * point[1]) / position_height)

  if confidence > threshold:
    cv2.circle(image2, (x, y), 3, (0,255,0), thickness = -1)
    cv2.putText(image2, "{}".format(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, .3, (0, 0, 255))
    cv2.putText(image2, '{}-{}'.format(point[0], point[1]), (x, y + 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,0,255)) # position of points
    points.append((x, y))
  else:
    points.append(None)

plt.figure(figsize = [14,10])
plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB));

def verify_arms_up(points):
  head, right_wrist, left_wrist = 0, 0, 0   #initializing them
  for i, point in enumerate(points):
    #print(i, point)
    if i==0:
      head = point[1]
    elif i==4:
      right_wrist = point[1]
    elif i==7:
      left_wrist = point[1]

    print(head, right_wrist, left_wrist)
    if right_wrist < head and left_wrist < head:
      return True
    else:
      return False

verify_arms_up(points)

# detecting gestures in videos(arms up)

video = '/content/drive/MyDrive/Computer Vision/Videos/gesture1.mp4'
capture = cv2.VideoCapture(video)
connected, frame = capture.read()

connected

frame.shape

result = '/content/drive/MyDrive/gesture1_finalResult.mp4'
save_video = cv2.VideoWriter(result, cv2.VideoWriter_fourcc(*'XVID'), 10, (frame.shape[1], frame.shape[0]))
# cv2.VideoWriter_fourcc(*'XVID') for codecc of video
# 10 for frame per second, if it's low the video is longer and we can process it easier, if it's higher it would be the same as original video

threshold = 0.1
while cv2.waitKey(1) < 0:
  connected, frame = capture.read()

  if not connected:
    break

  image_blob = cv2.dnn.blobFromImage(image = frame, scalefactor = 1.0 / 255, size = (256, 256))
  network.setInput(image_blob)
  output = network.forward()
  position_height = output.shape[2]
  position_width = output.shape[3]

  num_points = 15
  points = []
  for i in range(num_points):
    confidence_map = output[0, i, :, :]
    _, confidence, _, point = cv2.minMaxLoc(confidence_map)
    x = int((frame.shape[1] * point[0]) / position_width)
    y = int((frame.shape[0] * point[1]) / position_height)
    if confidence > threshold:
      cv2.circle(frame, (x, y), 5, (0,255,0), thickness = -1)
      cv2.putText(frame, "{}".format(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 255))
      points.append((x, y))
    else:
      points.append(None)

  for connection in point_connections:
    partA = connection[0]
    partB = connection[1]
    if points[partA] and points[partB]:
      cv2.line(frame, points[partA], points[partB], (255,0,0))

  if verify_arms_up(points) == True:
    cv2.putText(frame, 'Complete', (50,200), cv2.FONT_HERSHEY_COMPLEX, 3, (0,0,255))

  cv2_imshow(frame)
  save_video.write(frame)
save_video.release()

















