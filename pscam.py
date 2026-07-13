import threading
import time
from itertools import accumulate
from pathlib import Path
from types import EllipsisType
from typing import Sequence, TypeAlias, overload

import cv2
import numpy as np
from cv2.typing import MatLike


class VideoError(Exception):
    """Raise when error with fetching video from PS Camera"""
    pass

class PSCAM_VIEW:
    def __init__(self, frames: Sequence[MatLike], cvt: int | None = None):
        self._cvt = cvt
        self._frames = list(frames)

    @staticmethod
    def _check_side(side: int):
        if side < 0 or side > 1:
            raise IndexError("Side must be within 0, 1")

    @staticmethod
    def _check_frame(frame: int):
        if frame < 0 or frame > 3:
            raise IndexError("Frame must be within 0, 3")

    @overload
    def __getitem__(self, key: tuple[int, int | None] | int) -> MatLike: ...
    @overload
    def __getitem__(
        self, key: tuple[slice | EllipsisType, int | None] | slice | EllipsisType
    ) -> tuple[MatLike, MatLike]: ...
    @overload
    def __getitem__(self, key: tuple[int | None, slice | EllipsisType]) -> list[MatLike]: ...
    @overload
    def __getitem__(
        self, key: tuple[slice | EllipsisType, slice | EllipsisType]
    ) -> list[tuple[MatLike, MatLike]]: ...

    ViewIndex: TypeAlias = int | slice | None | EllipsisType
    ViewKey: TypeAlias = ViewIndex | tuple[ViewIndex, ViewIndex]

    def __getitem__(
        self, key: ViewKey
    ) -> (
        MatLike
        | tuple[MatLike, MatLike]
        | list[MatLike]
        | list[tuple[MatLike, MatLike]]
    ):
        if not isinstance(key, tuple):
            key = (key, None)
        key_side, key_frame = key

        if not key_side:
            key_side = 0
        elif isinstance(key_side, EllipsisType):
            key_side = slice(None)
        elif isinstance(key_side, int):
            self._check_side(key_side)

        if not key_frame:
            key_frame = 0
        elif isinstance(key_frame, EllipsisType):
            key_frame = slice(None)
        elif isinstance(key_frame, int):
            self._check_frame(key_frame)

        if isinstance(key_frame, int) and isinstance(key_side, int):
            frame = self._frames[key_side + key_frame * 2]
            if self._cvt:
                frame = cv2.cvtColor(frame, self._cvt)
            return frame
        if isinstance(key_side, slice) and isinstance(key_frame, slice):
            m, n, _ = key_side.indices(2)
            pairs = [
                (self._frames[m + i * 2], self._frames[n - 1 + i * 2])
                for i in key_frame.indices(4)
            ]
            if self._cvt:
                pairs = [
                    (cv2.cvtColor(pair[0], self._cvt), cv2.cvtColor(pair[1], self._cvt)[1])
                    for pair in pairs
                ]
            return pairs
        if isinstance(key_side, slice) and isinstance(key_frame, int):
            m, n, _ = key_side.indices(2)
            pair = (
                self._frames[m + key_frame * 2],
                self._frames[n - 1 + key_frame * 2],
            )
            if self._cvt:
                pair = (cv2.cvtColor(pair[0], self._cvt), cv2.cvtColor(pair[1], self._cvt) )
            return pair
        if isinstance(key_side, int) and not isinstance(key_frame, int):
            start, stop, step = key_frame.indices(4)
            frames = self._frames[key_side + start * 2 : key_side + stop * 2 : step * 2]
            if self._cvt:
                frames = [cv2.cvtColor(frame, self._cvt) for frame in frames]
            return frames
        raise KeyError("Invalid key.")


class PSCAM:
    CAPTURE_MODE_0 = 0
    CAPTURE_MODE_1 = 1
    CAPTURE_MODE_2 = 2
    _HEAD_PACK = [16, 32]

    def _buf_by_mode(self, mode: int):
        modes = {
            PSCAM.CAPTURE_MODE_0: (898, 200),
            PSCAM.CAPTURE_MODE_1: (1748, 408),
            PSCAM.CAPTURE_MODE_2: (3448, 808),
        }
        return modes.get(mode, (0, 0))

    def _res_by_mode(self, mode: int):
        modes = {
            PSCAM.CAPTURE_MODE_0: (320, 200),
            PSCAM.CAPTURE_MODE_1: (640, 400),
            PSCAM.CAPTURE_MODE_2: (1280, 800),
        }
        return modes.get(mode, (0, 0))

    def _pack_by_mode(self, mode: int):
        w = 320 * 2 ** self._scale_by_mode(mode)
        return [w // 4**i for i in range(4) for j in range(2)][:-1]

    def _res_by_mode_all(self, mode: int):
        scale = self._scale_by_mode(mode)
        w, h = 320 * 2**scale, 200 * 2**scale
        return [(w // 4**i, h // 2**i) for i in range(4) for j in range(2)]

    def _rows_size_by_mode(self, mode: int):
        return [
            self._res_by_mode(mode)[1] // 2**i for i in range(0, 4) for _ in range(2)
        ]

    def _scale_by_mode(self, mode: int):
        modes = {
            PSCAM.CAPTURE_MODE_0: 0,
            PSCAM.CAPTURE_MODE_1: 1,
            PSCAM.CAPTURE_MODE_2: 2,
        }
        return modes.get(mode, 0)

    def _config_uvc(self, mode: int, fps: int = 30):
        self._uvc.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        self._uvc.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
        self._uvc.set(cv2.CAP_PROP_FPS, fps)
        w, h = self._buf_by_mode(mode)
        self._uvc.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self._uvc.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    def _config_info(self, mode: int):
        self._w, self._h = self._res_by_mode(mode)
        self._res_all = self._res_by_mode_all(mode)
        self._buf_w, self._buf_h = self._buf_by_mode(mode)
        self._pack = list(accumulate(self._HEAD_PACK + self._pack_by_mode(mode)))
        self._rows_size = self._rows_size_by_mode(mode)
        self._rows_inter = (2, 2, 4, 4, 8, 8)
        self._mode = mode

    def _config_stereo(self):
        blockSize = 5
        self._n_disp = 256
        self.stereo = cv2.StereoSGBM.create(
            minDisparity=0,
            numDisparities=self._n_disp,
            blockSize=blockSize,
            P1=8 * blockSize**2,
            P2=24 * blockSize**2,
            disp12MaxDiff=1,
            uniquenessRatio=5,
            speckleWindowSize=75,
            speckleRange=2,
            preFilterCap=31,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

    def load_calib_data(self, path="./data/calib/stereo.npz"):
        file = Path(path)
        if file.is_file():
            data = np.load(path)
            self._map_l_x = data["map_l_x"]
            self._map_l_y = data["map_l_y"]
            self._map_r_x = data["map_r_x"]
            self._map_r_y = data["map_r_y"]
            self.calib = True
        else:
            print("Couldn't find calibration data.")
            self.calib = False

    def __init__(
        self, dev: int, mode: int = CAPTURE_MODE_0,
        stereo: cv2.StereoMatcher | None = None,
        #Use a NamedTuple?
        stereo_calib: tuple[ MatLike, MatLike, MatLike, MatLike ] | None = None
    ):
        self._uvc = cv2.VideoCapture(dev)
        self._config_uvc(mode)
        self._config_info(mode)
        self.calib = False
        self.load_calib_data()
        self._config_stereo()

    def start(self):
        self.stopped = False
        threading.Thread(target=self._capture_thread, daemon=True).start()
        return self

    def stop(self):
        self.stopped = True

    def _capture_thread(self):
        while not self.stopped:
            if not self._uvc.grab():
                self.stopped = True

    def _get_video(self):
        ret, data = self._uvc.retrieve()
        if not ret:
            self.stopped = True
            print("Stopped")
            return
        frames = data.reshape(self._buf_h, self._buf_w, 2)
        if self._mode != PSCAM.CAPTURE_MODE_0:
            frames = frames[:-8]
        frames = np.hsplit(frames, self._pack)[2:]
        frames[2:] = [
            np.roll(frame.reshape(-1, rows, res[0], 2), 1, axis=1).reshape(
                res[1], -1, 2
            )
            for frame, res, rows in zip(frames[2:], self._res_all[2:], self._rows_inter)
        ]
        return frames

    def _views(self, cvt: int | None = None):
        frames = self._get_video()
        if not frames:
            raise VideoError("No frames available.")
        return PSCAM_VIEW(frames, cvt)

    @property
    def frames_raw(self):
        return self._views()

    @property
    def frames_bgr(self):
        return self._views(cv2.COLOR_YUV2BGR_YUYV)

    @property
    def frames_gray(self):
        return self._views(cv2.COLOR_YUV2GRAY_YUYV)

    @property
    def frame_depth(self):
        frame_l, frame_r = self.frames_gray[:, 0]
        frame_l = cv2.remap(frame_l, self._map_l_x, self._map_l_y, cv2.INTER_LINEAR)
        frame_r = cv2.remap(frame_r, self._map_r_x, self._map_r_y, cv2.INTER_LINEAR)
        frame = self.stereo.compute(frame_l, frame_r).astype(np.float32) / 16.0
        cv2.normalize(
            frame, frame, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )
        return frame

    def _get_corners_calibration(self, raw_frame, size: tuple[int, int]):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        gray_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray_frame, size, None)
        if ret:
            corners = cv2.cornerSubPix(
                gray_frame, corners, (11, 11), (-1, -1), criteria
            )
            raw_frame = cv2.drawChessboardCorners(raw_frame, size, corners, ret)
        return (ret, corners, raw_frame)

    def _get_calibration_data(self, size: tuple[int, int], samples: int = 20):
        _o = np.zeros((size[0] * size[1], 3), np.float32)
        _o[:, :2] = np.mgrid[0 : size[0], 0 : size[1]].T.reshape(-1, 2)
        objp = [_o for _ in range(samples)]
        imgp_l = []
        imgp_r = []

        next = time.time() + 1
        for i in range(samples):
            while True:
                frame_l, frame_r = self.frames_gray[:, 0]
                ret_l, corn_l, frame_l = self._get_corners_calibration(frame_l, size)
                ret_r, corn_r, frame_r = self._get_corners_calibration(frame_r, size)
                frame = np.concat((frame_l, frame_r), axis=1)
                if time.time() >= next and ret_l and ret_r:
                    imgp_l.append(corn_l)
                    imgp_r.append(corn_r)
                    next = time.time() + 1
                    print(f"Capture #{i + 1}")
                    break
                cv2.imshow("Calibrating", frame)
                cv2.waitKey(10)
        cv2.destroyAllWindows()
        return (objp, imgp_l, imgp_r)

    def _get_intrinsics(
        self, objpoints: Sequence[MatLike], imgpoints: Sequence[MatLike]
    ):
        rms, mtx, dist, _, _ = cv2.calibrateCamera(
            objpoints,
            imgpoints,
            (self._w, self._h),
            None,
            None,
        )
        return (rms, mtx, dist)

    def _get_extrinsics(self, objp: Sequence[MatLike], imgp_l: Sequence[MatLike], imgp_r: Sequence[MatLike]):
        rms_l, mtx_l, dist_l = self._get_intrinsics(objp, imgp_l)
        rms_r, mtx_r, dist_r = self._get_intrinsics(objp, imgp_r)
        print(
            f"Left rms: {rms_l}. Right rms: {rms_r}. If either reprojection error is above 0.6, we suggest retrying the calibration sequence."
        )

        flags = cv2.CALIB_FIX_INTRINSIC
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        rms, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate(
            objp,
            imgp_l,
            imgp_r,
            mtx_l,
            dist_l,
            mtx_r,
            dist_r,
            (self._w, self._h),
            criteria = criteria,
            flags = flags,
        )
        print(
            f"Stereo rms is {rms}. If the reprojection error is above 0.5, we suggest retrying the calibration sequence."
        )
        rect_l, rect_r, proj_l, proj_r, _, _, _ = cv2.stereoRectify(
            mtx_l,
            dist_l,
            mtx_r,
            dist_r,
            (self._w, self._h),
            R,
            T,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0,
        )
        map_l_x, map_l_y = cv2.initUndistortRectifyMap(
            mtx_l, dist_l, rect_l, proj_l, (self._w, self._h), cv2.CV_32FC1
        )
        map_r_x, map_r_y = cv2.initUndistortRectifyMap(
            mtx_r, dist_r, rect_r, proj_r, (self._w, self._h), cv2.CV_32FC1
        )
        return (map_l_x, map_l_y, map_r_x, map_r_y)

    def calibrate(
        self, size: tuple[int, int], samples: int = 20, path="data/calib/stereo.npz"
    ):
        objp, imgp_l, imgp_r = self._get_calibration_data(size, samples)
        map_l_x, map_l_y, map_r_x, map_r_y = self._get_extrinsics(objp, imgp_l, imgp_r)

        np.savez(
            path, map_l_x=map_l_x, map_l_y=map_l_y, map_r_x=map_r_x, map_r_y=map_r_y
        )
