import numpy as np
import cv2
import sys

def main(arg):
    vid_capture = cv2.VideoCapture("20191119_1241_Cam_1_03_00.avi")

    transform_params = ((10, 15),
                        (5, 5),
                        (1, 1))
    frame_count = 0
    output_text = ""
    if arg is None:
        sequence_through = 10
    else:
        sequence_through = arg

    while (vid_capture.isOpened()):
        ret, frame = vid_capture.read()
        if ret == True:
            frame_count += 1
            if frame_count % sequence_through == 0:
                output_text += "" + str(frame_count) + "\t"
                for param in transform_params:
                    with_lines = frame.copy()
                    gray = cv2.cvtColor(with_lines, cv2.COLOR_BGR2GRAY)
                    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
                    lines = cv2.HoughLinesP(edges, param[0],
                                            param[1] * np.pi / 180, 100, minLineLength=50, maxLineGap=10)
                    for line in lines:
                        x1, y1, x2, y2 = line[0]
                        cv2.line(with_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    output_text += "" + str(len(lines)) + "\t"
        else:
            break

    with open("output_result.txt", "w") as f:
        f.write(output_text)

    vid_capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main(None)