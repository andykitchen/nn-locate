#!/usr/local/bin/python
#
# For GStreamer input, use OpenCV 3.0 ...
#   workon cv
#   python video_grab_images.py [image_pathname_prefix]

import cv2
import numpy as np
import sys

# --------------------------------------------------------------------------- #

grab_size = 64 / 2

image_pathname = 'image'
if len(sys.argv) == 2:
  image_pathname = sys.argv[1]
image_pathname = image_pathname + '_'

video_input = 0  # local camera

# gst-launch-1.0 udpsrc port=5000 ! application/x-rtp ! rtph264depay ! avdec_h264 ! videoconvert ! osxvideosink
# video_input = 'udpsrc port=5000 ! application/x-rtp ! rtph264depay ! avdec_h264 ! videoconvert ! appsink'

video_frame_rate = 1  # milliseconds

capture = cv2.VideoCapture(video_input)

# OpenCV 2.x
# video_size = (int(capture.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),
#               int(capture.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))
# video_fps  = (int(capture.get(cv2.cv.CV_CAP_PROP_FPS)))

# OpenCV 3.x
video_size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
              int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
video_fps  = (int(capture.get(cv2.CAP_PROP_FPS)))

print 'Video width, height:     ' + str(video_size)
print 'Video frames per second: ' + str(video_fps)

grab_count= 0

def grab(event, x, y, flags, parameter):
  if event == cv2.EVENT_LBUTTONDOWN:
    global grab_count, grab_image, grab_x, grab_y
    grab_image = frame[y-grab_size:y+grab_size, x-grab_size:x+grab_size]
    grab_x = x
    grab_y = y
    cv2.imwrite(image_pathname + '%0*d' % (4, grab_count) + '.jpg', grab_image);
    grab_count = grab_count + 1

cv2.namedWindow('Image')
cv2.namedWindow('Video')
cv2.setMouseCallback('Video', grab)

while (capture.isOpened()):
  r_okay, frame = capture.read()

  if r_okay == True:
    frame_output = frame.copy()

    if (grab_count > 0):
      frame_output[grab_y-grab_size:grab_y+grab_size,
                   grab_x-grab_size:grab_x+grab_size] = grab_image

      cv2.rectangle(frame_output,
        (grab_x-grab_size, grab_y-grab_size),
        (grab_x+grab_size, grab_y+grab_size),
        (0, 255, 255), 1)

    cv2.imshow('Video', frame_output)

  if cv2.waitKey(video_frame_rate) & 0xff == ord('q'):
    break

capture.release()
cv2.destroyAllWindows()
