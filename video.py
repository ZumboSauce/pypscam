import time

import cv2

from pscam import PSCAM
import numpy as np

if __name__ == "__main__":
    cam = PSCAM(0, PSCAM.CAPTURE_MODE_1)
    cam.start()

    time.sleep(1)

    while True:
        cv2.imshow("test", np.hstack( cam.view.bgr[:, 3] ) )
        cv2.waitKey(2)
