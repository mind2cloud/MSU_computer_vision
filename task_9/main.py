import numpy as np
import cv2
import os

def main():
    vid_capture = cv2.VideoCapture("Basler_for_calibration_piA640_5_5.mp4")
    ret, input = vid_capture.read()
    height, width, channels = input.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    coordinates_desk = []
    coordinates = []
    criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)
    calibration_grid = np.zeros((6 * 4, 3), np.float32)
    calibration_grid[:, :2] = np.mgrid[0:4, 0:6].T.reshape(-1, 2)

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
                corners_2 = cv2.cornerSubPix(gray_frame, corners_1, (11,11), (-1, -1), criteria)
                if [corners_2]:
                    coordinates.append(corners_2)
                else:
                    coordinates.append(corners_1)

                cv2.drawChessboardCorners(frame, (4, 6), corners_1, ret)
            print('Кадр {0:03d}'.format(frame_count))
        else:
            break

    vid_capture.release()

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(coordinates_desk[:30],
                                                       coordinates[:30],
                                                       size,
                                                       None,
                                                       None)

    # I. UNDISTORT

    vid_capture = cv2.VideoCapture("Basler_for_calibration_piA640_5_5.mp4")
    vdObj = cv2.VideoWriter("undistort_output_video.avi", fourcc, 10, (width, height))
    frame_count = 0
    while (vid_capture.isOpened()):
        frame_count += 1
        ret, frame = vid_capture.read()
        if ret == True:
            w, h = frame.shape[:2]
            new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (h, w))
            result = cv2.undistort(frame, mtx, dist, None, new_camera_matrix)

            vdObj.write(result)
            print('undistort Кадр {0:03d}'.format(frame_count))
            if frame_count % 5 == 0:
                path = '/Users/romankochnev/Desktop/task_9/undistort_output_result'
                filename = 'frame{0:03d}.jpg'.format(frame_count)
                cv2.imwrite(os.path.join(path, filename), result)
        else:
            break
    vid_capture.release()
    cv2.destroyAllWindows()

    # II. INITUNDISTORTRECTIFYMAP

    vid_capture = cv2.VideoCapture("Basler_for_calibration_piA640_5_5.mp4")
    vdObj = cv2.VideoWriter("initUndistortRectifyMap_output_video.avi", fourcc, 10, (width, height))
    frame_count = 0
    while (vid_capture.isOpened()):
        frame_count += 1
        ret, frame = vid_capture.read()
        if ret == True:
            w, h = frame.shape[:2]
            new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (h, w))
            rectify_map_1, rectify_map_2 = cv2.initUndistortRectifyMap(
                mtx, dist, None, new_camera_matrix,
                (frame.shape[1], frame.shape[0]), cv2.CV_32FC1)
            result = cv2.remap(frame, rectify_map_1, rectify_map_2, cv2.INTER_LINEAR)
            vdObj.write(result)
            print('UndistortRectifyMap Кадр {0:03d}'.format(frame_count))
            if frame_count % 5 == 0:
                path = '/Users/romankochnev/Desktop/task_9/initUndistortRectifyMap_output_result'
                filename = 'frame{0:03d}.jpg'.format(frame_count)
                cv2.imwrite(os.path.join(path, filename), result)
        else:
            break
    vid_capture.release()
    cv2.destroyAllWindows()




if __name__ == '__main__':
    main()