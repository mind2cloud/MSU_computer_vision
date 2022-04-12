import cv2
import os

def main():
    frame_count = 0
    vid_capture = cv2.VideoCapture("20191119_1241_Cam_1_03_00.avi")

    ret, input = vid_capture.read()
    height, width, channels = input.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vdObj = cv2.VideoWriter("output_video.avi", fourcc, 10, (width, height))

    while (vid_capture.isOpened()):
        ret, frame = vid_capture.read()
        if ret == True:
            print("frame # ", frame_count)
            frame_count += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            cornerImg = cv2.cornerHarris(gray, 2, 3, 0.04)
            normImg = cv2.normalize(cornerImg, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_32FC1, None)
            grayCornerImg = cv2.convertScaleAbs(normImg)
            mean = 0
            for x in range(grayCornerImg.shape[0]):
                for y in range(grayCornerImg.shape[1]):
                    mean += grayCornerImg[x, y]
            mean //= grayCornerImg.shape[0] * grayCornerImg.shape[1]
            for x in range(grayCornerImg.shape[0]):
                for y in range(grayCornerImg.shape[1]):
                    if grayCornerImg[x, y] > mean + 1:
                        cv2.circle(frame, (y, x), 3, (0, 0, 255))

            cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
            cv2.imshow("frame", frame)
            vdObj.write(frame)
            if frame_count % 5 == 0:
                path = '/Users/romankochnev/Desktop/task_4/output_result'
                filename = 'frame{0:03d}.jpg'.format(frame_count)
                cv2.imwrite(os.path.join(path, filename), frame)
        else:
            break

    vid_capture.release()
    vdObj.release()
    cv2.destroyAllWindows()




if __name__ == '__main__':
    main()