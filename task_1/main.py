import cv2
cv2.__version__

vid_capture = cv2.VideoCapture('20191119_1241_Cam_1_03_00.avi')

#----------------------
#       TASK 1
#----------------------
file_count = 0
while(vid_capture.isOpened()):
  ret, frame = vid_capture.read()
  if ret == True:
    if file_count % 10 == 0:
      print('Кадр {0:03d}'.format(file_count))
      filename = 'frame{0:03d}.jpg'.format(file_count)
      cv2.imwrite(filename, frame)
      key = cv2.waitKey(20)
    file_count += 1
  else:
    break


#----------------------
#       TASK 2
#----------------------
canny_values = [[50, 150], [30, 60], [150, 200]]

for values in canny_values:
  file_count = 0
  vid_capture = cv2.VideoCapture('20191119_1241_Cam_1_03_00.avi')
  
  while(vid_capture.isOpened()):
    ret, frame = vid_capture.read()
    if ret == True:
      if file_count % 5 == 0:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gaussian = cv2.GaussianBlur(frame_gray, (3, 3), cv2.BORDER_DEFAULT)
        canny_final = cv2.Canny(gaussian, values[0], values[1])

        print('Кадр {0:03d}'.format(file_count))
        filename = 'frame{0:03d}.jpg'.format(file_count)
        cv2.imwrite(filename, canny_final)

        key = cv2.waitKey(20)
      file_count += 1
    else:
      break