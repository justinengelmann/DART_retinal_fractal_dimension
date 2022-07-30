from functools import partial

import numpy as np
import torchvision.transforms as T
from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from .helpers import to_2tuple


def get_preprocessing(resolution=224, norm_mean_stds=None, resize=True,
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

    if transforms_backend == 'albumentations_if_available':
        transforms_backend = 'albumentations' if _check_albumentations_available() else 'torchvision'

    if transforms_backend == 'albumentations':
        import albumentations as A
        from albumentations.pytorch.transforms import ToTensorV2
        transforms = []
        if resize:
            transforms.append(A.Lambda(image=resize_fn, name='PIL_based_resize', always_apply=True))
        transforms.append(A.Normalize(norm_mean_stds[0], norm_mean_stds[1], always_apply=True))
        transforms.append(ToTensorV2())
        return DARTPreprocessingWrapper(preprocessing=A.Compose(transforms), transforms_backend='albumentations',
                                        resolution=resolution, resize=resize, norm_mean_stds=norm_mean_stds)

    elif transforms_backend == 'torchvision':
        transforms = []
        if resize:
            transforms.append(T.Lambda(resize_fn))
        transforms.append(T.Normalize(norm_mean_stds[0], norm_mean_stds[1]))
        transforms.append(T.ToTensor())
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
