import cv2
from cv2.typing import MatLike, TermCriteria
import numpy as np
from pscam import PSCAM

import time

def _get_corners_calibration(gray: MatLike, size: tuple[int, int], criteria):
    ret, corners = cv2.findChessboardCorners(gray, size, None)
    if ret:
        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria
        )
    return (ret, corners)

def _get_calibration_data(video: PSCAM, size: tuple[int, int], samples: int, criteria: TermCriteria):
    _o = np.zeros((size[0] * size[1], 3), np.float32)
    _o[:, :2] = np.mgrid[0 : size[0], 0 : size[1]].T.reshape(-1, 2)
    objp = [_o for _ in range(samples)]
    imgp_l = []
    imgp_r = []

    next = time.time() + 1
    for i in range(samples):
        while True:
            frame = video.view

            gray_l, gray_r = frame.gray[:]
            ret_l, corn_l = _get_corners_calibration(gray_l, size, criteria)
            ret_r, corn_r = _get_corners_calibration(gray_r, size, criteria)

            bgr_l, bgr_r = frame.bgr[:]
            bgr_l = cv2.drawChessboardCorners( bgr_l, size, corn_l, ret_l )
            bgr_r = cv2.drawChessboardCorners( bgr_r, size, corn_r, ret_r )
            frame = np.concat((bgr_l, bgr_r), axis=1)
            cv2.imshow("Calibrating", frame)

            if time.time() >= next and ret_l and ret_r:
                imgp_l.append(corn_l)
                imgp_r.append(corn_r)
                next = time.time() + 1
                print(f"Capture #{i + 1}")
                break

            cv2.waitKey(10)
    cv2.destroyAllWindows()
    return (objp, imgp_l, imgp_r)

def calibrate_stereo(
    video: PSCAM,
    size: tuple[int, int],
    samples: int = 30,
    rms_max: float = 1.0,
    subpix_criteria: TermCriteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
    stereo_cal_flags: int = cv2.CALIB_FIX_INTRINSIC,
    stereo_cal_criteria: TermCriteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
    stereo_rect_flags = cv2.CALIB_ZERO_DISPARITY
):
    objp, imgp_l, imgp_r = _get_calibration_data(video, size, samples, subpix_criteria)
    rms_l, mtx_l, dist_l, _, _ = cv2.calibrateCamera(
        objp,
        imgp_l,
        video.shape,
        None,
        None,
    )
    rms_r, mtx_r, dist_r, _, _ = cv2.calibrateCamera(
        objp,
        imgp_r,
        video.shape,
        None,
        None,
    )

    rms, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate(
        objp,
        imgp_l,
        imgp_r,
        mtx_l,
        dist_l,
        mtx_r,
        dist_r,
        video.shape,
        criteria = stereo_cal_criteria,
        flags = stereo_cal_flags,
    )
    print(
        f"Stereo rms is {rms}. If the reprojection error is above 0.5, we suggest retrying the calibration sequence."
    )
    rect_l, rect_r, proj_l, proj_r, _, _, _ = cv2.stereoRectify(
        mtx_l,
        dist_l,
        mtx_r,
        dist_r,
        video.shape,
        R,
        T,
        flags=stereo_rect_flags,
        alpha=0
    )
    map_l_x, map_l_y = cv2.initUndistortRectifyMap(
        mtx_l, dist_l, rect_l, proj_l, video.shape, cv2.CV_32FC1
    )
    map_r_x, map_r_y = cv2.initUndistortRectifyMap(
        mtx_r, dist_r, rect_r, proj_r, video.shape, cv2.CV_32FC1
    )

    return (map_l_x, map_l_y, map_r_x, map_r_y)
