import numpy as np
import cv2
import os

def main():
    vid_capture = cv2.VideoCapture("Basler_for_calibration_piA640_5_5.mp4")
    ret, input = vid_capture.read()
    height, width, channels = input.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vdObj = cv2.VideoWriter("output_video.avi", fourcc, 10, (width, height))

    coordinates_desk = []
    coordinates = []
    criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)
    calibration_grid = np.zeros((4 * 6, 3), np.float32)
    calibration_grid[:, :2] = np.mgrid[0:6, 0:4].T.reshape(-1, 2)

    frame_count = 0
    while (vid_capture.isOpened()):
        ret, frame = vid_capture.read()
        if ret == True:
            frame_count += 1

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            size = gray_frame.shape[::-1]
            ret, corners_1 = cv2.findChessboardCorners(gray_frame, (4, 6), None)

            if ret:
                coordinates_desk.append(calibration_grid)
                corners_2 = cv2.cornerSubPix(gray_frame, corners_1, (5, 5), (-1, -1), criteria)
                if [corners_2]:
                    coordinates.append(corners_2)
                else:
                    coordinates.append(corners_1)

                cv2.drawChessboardCorners(frame, (4, 6), corners_1, ret)

            vdObj.write(frame)
            print('Кадр {0:03d}'.format(frame_count))
            if frame_count % 5 == 0:
                path = '/Users/romankochnev/Desktop/task_8/output_result'
                filename = 'frame{0:03d}.jpg'.format(frame_count)
                cv2.imwrite(os.path.join(path, filename), frame)
        else:
            break

    vid_capture.release()
    cv2.destroyAllWindows()

    points_coordinates = '\n{}\n'.format(coordinates)
    with open("points_coordinates.txt", "a") as file:
        file.write(points_coordinates)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(coordinates_desk[:10],
                                                       coordinates[:10],
                                                       size, 
                                                       None, 
                                                       None)
    colibration_result = '\nret:{}\nmtx:{}\ndist:{}\nrvecs:{}\ntvecs:{}'.format(ret,
                                                                                mtx,
                                                                                dist,
                                                                                rvecs,
                                                                                tvecs)
    with open("colibration_result.txt", "a") as file:
            file.write(colibration_result)





if __name__ == '__main__':
    main()