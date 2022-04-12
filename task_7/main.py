import numpy as np
import cv2
import os

def main():
    vid_capture = cv2.VideoCapture(str("20191119_1241_Cam_1_03_00.avi"))
    ret, input = vid_capture.read()
    height, width, channels = input.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vdObj = cv2.VideoWriter("output_video.avi", fourcc, 10, (width, height))

    frame_count, tracks_old, p0_shape, p0r_shape = 0
    tracks = []
    track_len = 10

    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    while (vid_capture.isOpened()):
        frame_count += 1
        opened, frame = vid_capture.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if len(tracks) > 0:
            img_0, img_1 = prev_gray, frame_gray
            p_0 = np.float32([tr[-1] for tr in tracks]).reshape(-1, 1, 2)
            p_1, state, error = cv2.calcOpticalFlowPyrLK(img_0, img_1, p_0, None,
                                                   **lk_params)
            p_0r, state, error = cv2.calcOpticalFlowPyrLK(img_1, img_0, p_1, None,
                                                    **lk_params)
            d = abs(p_0 - p_0r).reshape(-1, 2).max(-1)
            cond = d < 5
            tracks_new = []
            num = 0
            for track, (x, y), good_flag in zip(tracks, p_1.reshape(-1, 2), cond):
                if not good_flag:
                    num += 1
                    continue
                track.append((x, y))
                if len(track) > track_len:
                    del track[0]
                tracks_new.append(track)
                cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)
            tracks = tracks_new
            cv2.polylines(frame, [np.int32(track) for track in tracks], False,
                          (0, 255, 0))

            logs = '\n{} {} {} {} {}\n'.format(frame_count, p_0.shape[0] - p0_shape,
                                               p_0r.shape[0] - p0r_shape, num,
                                               len(tracks_new) - tracks_old)
            with open("SIFT_logs.txt", "a") as file:
                file.write(logs)

            p0_shape = p_0.shape[0]
            p0r_shape = p_0r.shape[0]
            tracks_old = len(tracks_new)

        if frame_count % 1 == 0:
            mask = np.zeros_like(frame_gray)
            mask[:] = 255
            for x, y in [np.int32(tr[-1]) for tr in tracks]:
                cv2.circle(mask, (x, y), 5, 0, -1)
            points = calculate_SIFT(frame_gray)

            if points is not None:
                for x, y in np.float32(points).reshape(-1, 2):
                    tracks.append([(x, y)])

        prev_gray = frame_gray
        vdObj.write(frame)
        print('Кадр {0:03d}'.format(frame_count))
        if frame_count % 5 == 0:
            path = '/Users/romankochnev/Desktop/task_7'
            filename = 'frame{0:03d}.jpg'.format(frame_count)
            cv2.imwrite(os.path.join(path, filename), frame)
    vid_capture.release()
    vdObj.release()
    cv2.destroyAllWindows()

def calculate_SIFT(frame):
    model_image = cv2.imread('model_car.jpg', 0)
    sift = cv2.SIFT_create()

    keypoints_1, descriptors_1 = sift.detectAndCompute(model_image, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(frame, None)
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    comparisons = flann.knnMatch(descriptors_1, descriptors_2, k=2)
    comparisons_all = [[0, 0] for i in range(len(comparisons))]
    num = 0
    points = []
    for i, (m, n) in enumerate(comparisons):
        if m.distance < 0.6 * n.distance:
            num += 1
            comparisons_all[i] = [1, 0]
            pt2 = keypoints_2[m.trainIdx].pt
            points.append([int(pt2[0]), int(pt2[1])])
    points = np.array(points)
    points = np.expand_dims(points, axis=0)
    return points



if __name__ == '__main__':
    main()