import collections.abc
import logging
import os
from itertools import repeat
from urllib.error import HTTPError

import numpy as np
import torch
from torch.hub import download_url_to_file, urlparse, get_dir

from .available_models import model_cfgs

logger = logging.getLogger(__name__)


def _resolve_device(name) -> torch.device:
    if isinstance(name, torch.device):
        return name
    elif name == 'cuda_if_available':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        return torch.device(name)

def load_model(model_name='resnet18', use_jit=True, device='cuda_if_available', pbar=True, verbose=True) -> torch.nn.Module:
    device = _resolve_device(device)
    cfg = _get_cfg(model_name)
    if use_jit:
        url = cfg['jit_url']
    else:
        url = cfg['url']


    try:
        filepath = download_model(url=url, pbar=pbar, verbose=verbose)
    except HTTPError:
        raise ValueError(f'Could not download model {model_name} from {url}\n'
                         f'Check the releases https://github.com/justinengelmann/DART_retinal_fractal_dimension/releases/ '
                         f'and your internet connection.')

    model = torch.load(filepath, map_location=device)
    model.eval()

    if 'zero_tensor_test_output' in cfg:
        model.cpu()
        expected_output = torch.tensor(cfg['zero_tensor_test_output'])
        test_input = torch.zeros((1,) + cfg['input_size'], device='cpu')
        with torch.inference_mode():
            actual_output = model(test_input).cpu()[0]
            diff = actual_output.numpy() - expected_output.numpy()
            mean_diff = np.absolute(diff).mean()
        if not torch.allclose(actual_output, expected_output):
            if mean_diff < 1e-6:
                logger.warning(
                    f'Model {model_name} output diverges slightly from expect value (mean diff: {mean_diff}).\n'
                    f'Expected output: {expected_output}\nActual output: {actual_output}\nDiff: {diff}')
            else:
                raise ValueError(f'Model {model_name} output diverges from expect value (mean diff: {mean_diff}).\n'
                                 f'Expected output: {expected_output}\nActual output: {actual_output}\nDiff: {diff}')
        else:
            if verbose:
                logger.info(f'Model {model_name} output matches expect value (mean diff: {mean_diff}).\n'
                            f'Expected output: {expected_output}\nActual output: {actual_output}\nDiff: {diff}')
        model.to(device)
    return model


def download_model(url, hash_prefix=None, pbar=True, verbose=False):
    filename = urlparse(url).path.split('/')[-1]

    torch_hub_dir = get_dir()
    dart_dir = os.path.join(torch_hub_dir, 'checkpoints', 'dart')
    os.makedirs(dart_dir, exist_ok=True)

    file_path = os.path.join(dart_dir, filename)

    if not os.path.exists(file_path):
        if verbose:
            logging.info(f'Downloading {url} to {file_path}.')
        download_url_to_file(url, file_path, hash_prefix=hash_prefix, progress=pbar)

    return file_path


def _get_cfg(model_name) -> dict:
    model_name = model_name.lower()
    if model_name not in model_cfgs:
        raise ValueError(f'Unknown model {model_name}, currently supported models are: {list(model_cfgs.keys())}')
    return model_cfgs[model_name]


# From PyTorch internals, inspired by Ross Wightman
# https://github.com/rwightman/pytorch-image-models/blob/7430a85d07a7f335e18c2225fda4a5e7b60b995c/timm/models/layers/helpers.py#L19
def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    return tuple(repeat(x, 2))
