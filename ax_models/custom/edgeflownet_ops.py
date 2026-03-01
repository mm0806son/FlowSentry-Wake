import numpy as np
import cv2
import torch

from axelera import types
from axelera.app.operators import AxOperator
from axelera.app.operators.base import PreprocessOperator


class EdgeFlowNetPreprocess(PreprocessOperator):
    width: int = 1024
    height: int = 576
    scale: float = 1.0 / 255.0
    mean0: float = 0.0
    mean1: float = 0.0
    mean2: float = 0.0
    std0: float = 1.0
    std1: float = 1.0
    std2: float = 1.0
    quant_scale: float = 0.003921568859
    quant_zeropoint: int = -128
    swap_rb: bool = False

    def _post_init(self):
        super()._post_init()
        self._prev = None

    def build_gst(self, gst, stream_idx: str):
        gst.videoconvert()
        gst.capsfilter(caps='video/x-raw,format=RGB')
        options = (
            f"width:{self.width};"
            f"height:{self.height};"
            f"scale:{self.scale};"
            f"mean0:{self.mean0};mean1:{self.mean1};mean2:{self.mean2};"
            f"std0:{self.std0};std1:{self.std1};std2:{self.std2};"
            f"quant_scale:{self.quant_scale};"
            f"quant_zeropoint:{self.quant_zeropoint};"
            f"swap_rb:{int(self.swap_rb)}"
        )
        gst.axtransform(lib="libedgeflownet_pre.so", options=options)

    def exec_torch(self, image: types.Image) -> torch.Tensor:
        img = image.asarray(types.ColorFormat.RGB)
        if img is None:
            raise ValueError("EdgeFlowNetPreprocess: input image is None")
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"EdgeFlowNetPreprocess: unexpected shape {img.shape}")

        if self.width > 0 and self.height > 0:
            img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

        img = img.astype(np.float32) * self.scale
        if self.swap_rb:
            img = img[..., ::-1]

        if self._prev is None:
            self._prev = img.copy()

        prev = self._prev
        self._prev = img.copy()

        if (self.mean0, self.mean1, self.mean2) != (0.0, 0.0, 0.0) or (
            self.std0,
            self.std1,
            self.std2,
        ) != (1.0, 1.0, 1.0):
            mean = np.array([self.mean0, self.mean1, self.mean2], dtype=np.float32)
            std = np.array([self.std0, self.std1, self.std2], dtype=np.float32)
            prev = (prev - mean) / std
            img = (img - mean) / std

        combined = np.concatenate([prev, img], axis=-1)
        combined = np.transpose(combined, (2, 0, 1))
        return torch.from_numpy(np.ascontiguousarray(combined))


class EdgeFlowNetPostprocess(AxOperator):
    max_flow: float = 50.0
    gamma: float = 1.0
    out_width: int = 0
    out_height: int = 0
    meta_key: str = "opticalflow"

    def build_gst(self, gst, stream_idx: str):
        options = (
            f"max_flow:{self.max_flow};"
            f"gamma:{self.gamma};"
            f"out_width:{self.out_width};"
            f"out_height:{self.out_height};"
            f"meta_key:{self.meta_key}"
        )
        gst.decode_muxer(lib="libedgeflownet_post.so", options=options)

    def exec_torch(self, image, predict, axmeta):
        return image, predict, axmeta


class EdgeFlowNetDraw(AxOperator):
    # CPU-only draw path; force CPU pipeline to avoid CL/VAAPI buffers.
    _force_cpu_pipeline = True

    def build_gst(self, gst, stream_idx: str):
        if getattr(gst, "building_axinference", False):
            gst.finish_axinference()
        gst.axtransform(lib="libtransform_colorconvert.so", options="format:rgb")
        gst.capsfilter(caps='video/x-raw,format=RGB')
        gst.axinplace(lib="libinplace_draw.so", mode="write")

    def exec_torch(self, image, predict, axmeta):
        return image, predict, axmeta
