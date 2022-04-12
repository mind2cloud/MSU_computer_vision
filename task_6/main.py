import numpy as np
import cv2
import os

def main():
    vid_capture = cv2.VideoCapture("20191119_1241_Cam_1_03_00.avi")
    image = cv2.imread("model_car.jpg", cv2.IMREAD_COLOR)
    model_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    ret, input = vid_capture.read()
    height, width, channels = input.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vdObj = cv2.VideoWriter("output_video.avi", fourcc, 10, (width, height))

    frame_count = 0
    while (vid_capture.isOpened()):
        ret, frame = vid_capture.read()
        if ret == True:
            frame_count += 1
            sift = cv2.SIFT_create()
            keypoints_1, descriptors_1 = sift.detectAndCompute(model_image, None)
            keypoints_2, descriptors_2 = sift.detectAndCompute(frame, None)

            FLANN_INDEX_KDTREE = 0
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            search_params = dict(checks = 50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            comparisons = flann.knnMatch(descriptors_1, descriptors_2, k = 2)
            comparisons_all = [[0, 0] for i in range(len(comparisons))]
            num = 0
            for i, (m, n) in enumerate(comparisons):
                if m.distance < 0.6 * n.distance:
                    num += 1
                    comparisons_all[i] = [1, 0]

            draw_params = dict(matchColor=(0, 255, 0),
                               singlePointColor=(255, 0, 0),
                               matchesMask=comparisons_all,
                               flags=0)
            img3 = cv2.drawMatchesKnn(model_image, keypoints_1, frame, keypoints_2, comparisons, None, **draw_params)
            res = cv2.resize(img3, (1600, 1200), interpolation=cv2.INTER_CUBIC)

            logs = '\n{} {} {} {}\n'.format(frame_count, len(keypoints_2), len(keypoints_1), num)
            with open("SIFT_logs.txt", "a") as file:
                file.write(logs)

            video_image = res
            vdObj.write(video_image)
            print('Кадр {0:03d}'.format(frame_count))
            if frame_count % 5 == 0:
                path = '/Users/romankochnev/Desktop/task_6/output_result'
                filename = 'frame{0:03d}.jpg'.format(frame_count)
                cv2.imwrite(os.path.join(path, filename), video_image)
        else:
            break
    vid_capture.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()