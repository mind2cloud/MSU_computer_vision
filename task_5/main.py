import numpy as np
import cv2
import os

def main():
    vid_capture = cv2.VideoCapture("20191119_1241_Cam_1_03_00.avi")

    ret, input = vid_capture.read()
    height, width, channels = input.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vdObj = cv2.VideoWriter("output_video.avi", fourcc, 10, (width, height))

    frame_count = 0
    while (vid_capture.isOpened()):
        ret, frame = vid_capture.read()
        if ret == True:
            frame_count += 1
            frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sift = cv2.SIFT_create()
            kpnts = sift.detect(frameGray, None)
            imgWithKeypoints = cv2.drawKeypoints(frameGray, kpnts, frame, (255, 255, 0),
                                                 flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            vdObj.write(imgWithKeypoints)
            cv2.imshow("image",imgWithKeypoints)
            print('Кадр {0:03d}'.format(frame_count))
            if frame_count % 5 == 0:
                path = '/Users/romankochnev/Desktop/task_5/output_result'
                filename = 'frame{0:03d}.jpg'.format(frame_count)
                cv2.imwrite(os.path.join(path, filename), frame)
        else:
            break

    vid_capture.release()
    vdObj.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()