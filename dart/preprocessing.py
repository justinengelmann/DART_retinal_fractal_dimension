from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as T
from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from .helpers import to_2tuple


def get_preprocessing(resolution=224, norm_mean_stds=None,
                      resize=True,
                      crop_black_borders=False, crop_threshold=20,
                      transforms_backend='albumentations_if_available'):
    """
    This aims to replicate the preprocessing used by the DART model faithfully,
    even if the lambda resizing might not be the fastest option.
    The model was trained with albumentations, but to keep the dependencies low, we also offer torchvision transforms.
    There might be minimal differences between the two, in the order of 10^-7 for the transformed images but no
    discernible difference in the model's output.
    """
    resolution = to_2tuple(resolution)
    norm_mean_stds = (IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD) if norm_mean_stds is None else norm_mean_stds
    if resize:
        resize_fn = partial(_resize_img_with_PIL, size=resolution, interpolation=Image.BILINEAR)
    if crop_black_borders:
        crop_fn = partial(_crop_black_borders, threshold=crop_threshold)

    if transforms_backend == 'albumentations_if_available':
        transforms_backend = 'albumentations' if _check_albumentations_available() else 'torchvision'

    if transforms_backend == 'albumentations':
        import albumentations as A
        from albumentations.pytorch.transforms import ToTensorV2
        transforms = []
        if crop_black_borders:
            transforms.append(A.Lambda(image=crop_fn, name='crop_black_borders', always_apply=True))
        if resize:
            transforms.append(A.Lambda(image=resize_fn, name='PIL_based_resize', always_apply=True))
        transforms.append(A.Normalize(norm_mean_stds[0], norm_mean_stds[1], always_apply=True))
        transforms.append(ToTensorV2())
        return DARTPreprocessingWrapper(preprocessing=A.Compose(transforms), transforms_backend='albumentations',
                                        resolution=resolution, resize=resize, norm_mean_stds=norm_mean_stds)

    elif transforms_backend == 'torchvision':
        transforms = []
        if crop_black_borders:
            transforms.append(T.Lambda(crop_fn))
        if resize:
            transforms.append(T.Lambda(resize_fn))
            transforms.append(T.Lambda(lambda x: Image.fromarray(x)))
        transforms.append(T.ToTensor())
        transforms.append(T.Normalize(norm_mean_stds[0], norm_mean_stds[1]))
        return DARTPreprocessingWrapper(preprocessing=T.Compose(transforms), transforms_backend='torchvision',
                                        resolution=resolution, resize=resize, norm_mean_stds=norm_mean_stds)


class DARTPreprocessingWrapper:
    def __init__(self, preprocessing, transforms_backend, resolution=None, resize=None, norm_mean_stds=None):
        self.preprocessing = preprocessing
        self.transforms_backend = transforms_backend
        assert transforms_backend in ['albumentations', 'torchvision']

        self.resolution = resolution
        self.resize = resize
        self.norm_mean_stds = norm_mean_stds

    def __call__(self, img):
        if self.transforms_backend == 'albumentations':
            # albumentations doesn't accept PIL
            if not isinstance(img, np.ndarray):
                img = np.array(img)
            return self.preprocessing(image=img)['image']
        else:
            return self.preprocessing(img)

    def __repr__(self):
        out = f'DARTPreprocessingWrapper(transforms_backend={self.transforms_backend}'
        if self.resolution is not None:
            out += f',resolution={self.resolution}'
        if self.resize is not None:
            out += f', resize={self.resize}'
            if not self.resize:
                out += '(Note: we expect that the input image is already resized to the desired resolution.)'
        if self.norm_mean_stds is not None:
            out += f', norm_mean_stds={self.norm_mean_stds}'
        out += ')'
        return out


def _check_albumentations_available():
    try:
        import albumentations as A
        from albumentations.pytorch.transforms import ToTensorV2
        return True
    except ImportError:
        return False


def _resize_img_with_PIL(img, size, interpolation=Image.ANTIALIAS, **kwargs) -> np.ndarray:
    # albumentations might force unwanted kwargs on us, so that's why we accept them here
    resolution = to_2tuple(size)
    if not isinstance(img, Image.Image):
        # convert to PIL.Image first
        img = Image.fromarray(img)
    return np.array(img.resize(size=resolution, resample=interpolation))


def _crop_black_borders(img, threshold=20):
    """
    Crops out the black borders of a retinal fundus image, and returns the cropped image.
    Attempts to return a square image.
    The border are rectangular image edges that are black and not part of the fundus.
    threshold is the minimum mean intensity of a pixel to be counted as non-black.
    """

    # convert to PIL.Image first
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)

    # make a black and white copy
    img_np = np.array(img).mean(axis=2)
    img_np = (img_np > threshold).astype(np.uint8)

    # find top edge
    top_edge = np.where(img_np.sum(axis=1) > 0)[0][0]
    # find bottom edge
    bottom_edge = np.where(img_np.sum(axis=1) > 0)[0][-1]
    # find left edge
    left_edge = np.where(img_np.sum(axis=0) > 0)[0][0]
    # find right edge
    right_edge = np.where(img_np.sum(axis=0) > 0)[0][-1]

    # add 5 pixels to the edges to avoid cropping out the edges, but check that this doesn't go out of bounds
    top_edge = max(0, top_edge - 5)
    bottom_edge = min(img.size[1], bottom_edge + 5)
    left_edge = max(0, left_edge - 5)
    right_edge = min(img.size[0], right_edge + 5)

    x_dim = right_edge - left_edge
    y_dim = bottom_edge - top_edge

    # extend shorter dimension to yield a square image
    x_minus_y_dim = x_dim - y_dim
    extend_by = np.abs(x_minus_y_dim) // 2
    if x_minus_y_dim > 0:
        top_edge = max(0, top_edge - extend_by)
        bottom_edge = min(img.size[1], bottom_edge + extend_by)
    elif x_minus_y_dim < 0:
        left_edge = max(0, left_edge - extend_by)
        right_edge = min(img.size[0], right_edge + extend_by)
    else:
        # nothing to do, image is already square
        pass

    # crop out the black borders
    return img.crop((left_edge, top_edge, right_edge, bottom_edge))
