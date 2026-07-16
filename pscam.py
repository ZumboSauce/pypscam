import threading
from itertools import accumulate
from types import EllipsisType
from typing import Sequence, TypeAlias, overload

import cv2
import numpy as np
from cv2.typing import MatLike


class PSCamError(Exception): ...

class PSCam_Stereo():
    def __init__( self,
        stereo: cv2.StereoMatcher,
        calibration: tuple[ MatLike, MatLike, MatLike, MatLike ],
        normalize: bool = True,
        interpolation: int = cv2.INTER_LINEAR,
        alpha: float = 0,
        beta: float = 255,
        norm_type: int = cv2.NORM_MINMAX,
        dtype: int = cv2.CV_8U,
        mask: MatLike | None = None
    ):
        self._stereo = stereo
        self._l_x, self._l_y, self._r_x, self._r_y = calibration
        self._n = normalize
        self._i = interpolation
        self._a = alpha
        self._b = beta
        self._n_t = norm_type
        self._dt = dtype
        self._mask = mask

    def compute_depth( self, view: tuple[MatLike, MatLike] ):
        frame_l = cv2.remap(view[0], self._l_x, self._l_y, self._i)
        frame_r = cv2.remap(view[1], self._r_x, self._r_y, self._i)
        depth = self._stereo.compute(frame_l, frame_r).astype(np.float32) / 16.0
        if self._n:
            cv2.normalize(
                depth, depth, alpha=self._a, beta=self._b, norm_type=self._n_t, dtype=self._dt, mask=self._mask
            )
        return depth

class PSView:
    def __init__(self, frames: Sequence[MatLike], cvt: int | None = None):
        self._cvt = cvt
        self._frames = list(frames)

    @property
    def raw(self):
        return PSView( self._frames, None )

    @property
    def gray(self):
        return PSView( self._frames, cv2.COLOR_YUV2GRAY_YUYV )

    @property
    def bgr(self):
        return PSView( self._frames, cv2.COLOR_YUV2BGR_YUYV )

    def cvt(self, cvt: int | None):
        return PSView( self._frames, cvt ) if cvt != self._cvt else self

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

    def __init__(
        self,
        dev: int,
        mode: int = CAPTURE_MODE_0,
        stereo: PSCam_Stereo | None = None
    ):
        self._uvc = cv2.VideoCapture(dev)
        self._mode = mode
        self._stereo = stereo
        self._config_uvc(mode)
        self._config_info(mode)

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

    @property
    def view(self):
        frames = self._get_video()
        if not frames:
            raise PSCamError("Couldn't fetch frames from device.")
        return PSView(frames)

    @property
    def frame_depth(self):
        if self._stereo:
            return self._stereo.compute_depth( self.view.gray[:] )
        raise PSCamError( "Stereo matcher was never initialized." )

    @property
    def shape(self):
        return (self._h, self._w)
