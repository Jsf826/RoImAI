import sys
from typing import Tuple, List, Union, Optional

import numpy as np
import torch
from PIL import Image
from fvcore.transforms import NoOpTransform, Transform, TransformList
import torch.nn.functional as F
from torchvision.transforms import transforms as tvt

from detectron2.data import transforms as T
from detectron2.data.transforms import Augmentation, ResizeTransform, ResizeShortestEdge, AugmentationList

__all__ = ["MultiImageResizeShortestEdge", "MulAugInput", "rand_gaussian_blur", "rand_color_augmentation"]


class MultiImageResizeTransform(Transform):
    """
    Resize the image to a target size.
    """

    def __init__(self, h, w, new_h, new_w, interp=None):
        """
        Args:
            h, w (int): original image size
            new_h, new_w (int): new image size
            interp: PIL interpolation methods, defaults to bilinear.
        """
        # TODO decide on PIL vs opencv
        super().__init__()
        if interp is None:
            interp = Image.BILINEAR
        self._set_attributes(locals())

    def apply_image(self, img, interp=None):
        assert img.shape[1:3] == (self.h, self.w)
        assert len(img.shape) == 4
        interp_method = interp if interp is not None else self.interp

        if img.dtype == np.uint8:

            pil_image1 = Image.fromarray(img[0][:, :, ::-1])
            pil_image2 = Image.fromarray(img[1][:, :, ::-1])
            pil_image1 = pil_image1.resize((self.new_w, self.new_h), interp_method)
            pil_image2 = pil_image2.resize((self.new_w, self.new_h), interp_method)
            ret1 = np.asarray(pil_image1)[:, :, ::-1]
            ret2 = np.asarray(pil_image2)[:, :, ::-1]
            ret = np.stack((ret1, ret2), axis=0)
        else:
            raise Exception("图片数据类型不为np.uint8!")

        return ret

    def apply_coords(self, coords):
        coords[:, 0] = coords[:, 0] * (self.new_w * 1.0 / self.w)
        coords[:, 1] = coords[:, 1] * (self.new_h * 1.0 / self.h)
        return coords

    def apply_segmentation(self, segmentation):
        segmentation = self.apply_image(segmentation, interp=Image.NEAREST)
        return segmentation

    def inverse(self):
        return ResizeTransform(self.new_h, self.new_w, self.h, self.w, self.interp)


class MultiImageResizeShortestEdge(Augmentation):
    """
    Resize the image while keeping the aspect ratio unchanged.
    It attempts to scale the shorter edge to the given `short_edge_length`,
    as long as the longer edge does not exceed `max_size`.
    If `max_size` is reached, then downscale so that the longer edge does not exceed max_size.
    """

    @torch.jit.unused
    def __init__(
            self, short_edge_length, max_size=sys.maxsize, sample_style="range", interp=Image.BILINEAR
    ):
        """
        Args:
            short_edge_length (list[int]): If ``sample_style=="range"``,
                a [min, max] interval from which to sample the shortest edge length.
                If ``sample_style=="choice"``, a list of shortest edge lengths to sample from.
            max_size (int): maximum allowed longest edge length.
            sample_style (str): either "range" or "choice".
        """
        super().__init__()
        assert sample_style in ["range", "choice"], sample_style

        self.is_range = sample_style == "range"
        if isinstance(short_edge_length, int):
            short_edge_length = (short_edge_length, short_edge_length)
        if self.is_range:
            assert len(short_edge_length) == 2, (
                "short_edge_length must be two values using 'range' sample style."
                f" Got {short_edge_length}!"
            )
        self._init(locals())

    @torch.jit.unused
    def get_transform(self, image):
        assert len(image.shape) == 4, "image维度必须为4"
        h, w = image.shape[1:3]
        if self.is_range:
            size = np.random.randint(self.short_edge_length[0], self.short_edge_length[1] + 1)
        else:
            size = np.random.choice(self.short_edge_length)
        if size == 0:
            return NoOpTransform()

        newh, neww = ResizeShortestEdge.get_output_shape(h, w, size, self.max_size)
        return MultiImageResizeTransform(h, w, newh, neww, self.interp)

    @staticmethod
    def get_output_shape(
            oldh: int, oldw: int, short_edge_length: int, max_size: int
    ) -> Tuple[int, int]:
        """
        Compute the output size given input size and target short edge length.
        """
        h, w = oldh, oldw
        size = short_edge_length * 1.0
        scale = size / min(h, w)
        if h < w:
            newh, neww = size, scale * w
        else:
            newh, neww = scale * h, size
        if max(newh, neww) > max_size:
            scale = max_size * 1.0 / max(newh, neww)
            newh = newh * scale
            neww = neww * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)


def rand_color_augmentation(image1, image2, p=0.8, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1):
    """
    同时对两张图像应用相同的颜色增强
    :param image1: 第一张图像
    :param image2: 第二张图像
    :param p: 应用增强的概率
    :return: 增强后的两张图像
    """
    # 如果没有触发增强，则不做任何处理，直接返回原图
    aug_image1, aug_image2 = image1, image2

    # 生成一个随机值，决定是否进行增强
    if np.random.rand() < p:
        # 触发颜色增强时，应用相同的颜色增强到两张图像
        pil_image1 = Image.fromarray(image1[:, :, ::-1])  # 转换为 PIL 图像
        pil_image2 = Image.fromarray(image2[:, :, ::-1])  # 转换为 PIL 图像

        trans_color_aug = tvt.Compose([
            tvt.ColorJitter(brightness, contrast, saturation, hue)  # 在这里设置颜色增强的参数
        ])

        # 应用增强
        aug_image1 = np.asarray(trans_color_aug(pil_image1))[:, :, ::-1]  # 转回 numpy，并恢复通道顺序
        aug_image2 = np.asarray(trans_color_aug(pil_image2))[:, :, ::-1]  # 转回 numpy，并恢复通道顺序

    return aug_image1, aug_image2


def rand_gaussian_blur(image1, image2, p=0.5, kernel_size=3, sigma=0.5):
    """
    对图像应用高斯模糊。
    :param image: 输入的图像，应该是 numpy 数组格式
    :param kernel_size: 高斯模糊的内核大小（必须是奇数）
    :param sigma: 高斯模糊的标准差
    :return: 应用高斯模糊后的图像
    """
    # 如果没有触发增强，则不做任何处理，直接返回原图
    aug_image1, aug_image2 = image1, image2

    # 生成一个随机值，决定是否进行增强
    if np.random.rand() < p:
        # 将 numpy 数组转换为 PIL 图像
        pil_image1 = Image.fromarray(image1[:, :, ::-1])  # 转换为 PIL 图像
        pil_image2 = Image.fromarray(image2[:, :, ::-1])  # 转换为 PIL 图像

        # 创建 GaussianBlur 转换
        gaussian_blur = tvt.GaussianBlur(kernel_size, sigma=sigma)

        # 应用增强
        aug_image1 = np.asarray(gaussian_blur(pil_image1))[:, :, ::-1]  # 转回 numpy，并恢复通道顺序
        aug_image2 = np.asarray(gaussian_blur(pil_image2))[:, :, ::-1]  # 转回 numpy，并恢复通道顺序

    return aug_image1, aug_image2


class MulAugInput:
    """
    Input that can be used with :meth:`Augmentation.__call__`.
    This is a standard implementation for the majority of use cases.
    This class provides the standard attributes **"image", "boxes", "sem_seg"**
    defined in :meth:`__init__` and they may be needed by different augmentations.
    Most augmentation policies do not need attributes beyond these three.

    After applying augmentations to these attributes (using :meth:`AugInput.transform`),
    the returned transforms can then be used to transform other data structures that users have.

    Examples:
    ::
        input = AugInput(image, boxes=boxes)
        tfms = augmentation(input)
        transformed_image = input.image
        transformed_boxes = input.boxes
        transformed_other_data = tfms.apply_other(other_data)

    An extended project that works with new data types may implement augmentation policies
    that need other inputs. An algorithm may need to transform inputs in a way different
    from the standard approach defined in this class. In those rare situations, users can
    implement a class similar to this class, that satify the following condition:

    * The input must provide access to these data in the form of attribute access
      (``getattr``).  For example, if an :class:`Augmentation` to be applied needs "image"
      and "sem_seg" arguments, its input must have the attribute "image" and "sem_seg".
    * The input must have a ``transform(tfm: Transform) -> None`` method which
      in-place transforms all its attributes.
    """

    # TODO maybe should support more builtin data types here
    def __init__(
            self,
            image: np.ndarray,
            *,
            boxes: Optional[np.ndarray] = None,
            sem_seg: Optional[np.ndarray] = None,
    ):
        """
        Args:
            image (ndarray): (H,W) or (H,W,C) ndarray of type uint8 in range [0, 255], or
                floating point in range [0, 1] or [0, 255]. The meaning of C is up
                to users.
            boxes (ndarray or None): Nx4 float32 boxes in XYXY_ABS mode
            sem_seg (ndarray or None): HxW uint8 semantic segmentation mask. Each element
                is an integer label of pixel.
        """
        assert len(image.shape) == 4
        self.image = image
        self.boxes = boxes
        self.sem_seg = sem_seg

    def transform(self, tfm: Transform) -> None:
        """
        In-place transform all attributes of this class.

        By "in-place", it means after calling this method, accessing an attribute such
        as ``self.image`` will return transformed data.
        """
        self.image = tfm.apply_image(self.image)
        if self.boxes is not None:
            self.boxes = tfm.apply_box(self.boxes)
        if self.sem_seg is not None:
            self.sem_seg = tfm.apply_segmentation(self.sem_seg)

    def apply_augmentations(
            self, augmentations: List[Union[Augmentation, Transform]]
    ) -> TransformList:
        """
        Equivalent of ``AugmentationList(augmentations)(self)``
        """
        return AugmentationList(augmentations)(self)
