import time

import cv2

from pscam import PSCAM

if __name__ == "__main__":
    cam = PSCAM(0, PSCAM.CAPTURE_MODE_2)
    cam.start()

    time.sleep(1)

    while True:
        cv2.imshow("test", cam.frames_bgr[0, 3] )
        cv2.waitKey(2)
