import cv2
import numpy as np
import math
import os

def main():
    vid_capture = cv2.VideoCapture("20191119_1241_Cam_1_03_00.avi")
    frame_count = 0
    while (vid_capture.isOpened()):
        ret, frame = vid_capture.read()
        if ret == True:
            frame_count += 1
            # for each 5th frame
            if frame_count % 5 == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                gray = cv2.GaussianBlur(gray, (3, 3), 0)
                canny = cv2.Canny(gray, 50, 100, 3)
                lines = cv2.HoughLinesP(canny, rho=5, theta=np.pi / 180,
                                        threshold=80, minLineLength=30, maxLineGap=5)
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    if math.fabs(x1 - x2) <= 5.0 or math.fabs(y1 - y2) <= 5.0:
                        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    else:
                        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                print('Кадр {0:03d}'.format(frame_count))
                path = '/Users/romankochnev/Desktop/task_3/output_result'
                filename = 'frame{0:03d}.jpg'.format(frame_count)
                cv2.imwrite(os.path.join(path, filename), frame)
        else:
            break
    vid_capture.release()
    cv2.destroyAllWindows()





if __name__ == '__main__':
    main()