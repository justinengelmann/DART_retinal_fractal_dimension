import torch
from torch import nn

from .helpers import load_model, _get_cfg, _resolve_device
from .postprocessing import get_postprocessing
from .preprocessing import get_preprocessing



def get_model_and_processing(model_name='resnet18', device='cuda_if_available', use_jit=True,
                             resize_images=True, crop_black_borders=False, crop_threshold=20,
                             preprocessing_backend='albumentations_if_available',
                             loading_verbose=True, loading_pbar=True):
    cfg = _get_cfg(model_name=model_name)

    preprocessing = get_preprocessing(resolution=cfg['input_size'][1:],
                                      norm_mean_stds=cfg['norm_mean_stds'],
                                      resize=resize_images,
                                      crop_black_borders=crop_black_borders,
                                      crop_threshold=crop_threshold,

                                      transforms_backend=preprocessing_backend)

    postprocessing = get_postprocessing(cfg=cfg)

    model = load_model(model_name=model_name, device=device, use_jit=use_jit, pbar=loading_pbar, verbose=loading_verbose)

    return {'model': model, 'preprocessing': preprocessing, 'postprocessing': postprocessing, 'cfg': cfg}


def get_inference_pipeline(model_name='resnet18', device='cuda_if_available', use_jit=True,
                           resize_images=True, crop_black_borders=False, crop_threshold=20,
                           preprocessing_backend='albumentations_if_available',
                           loading_verbose=True, loading_pbar=True):
    model_and_processing = get_model_and_processing(model_name=model_name, device=device, use_jit=use_jit,
                                                    resize_images=resize_images,
                                                    crop_black_borders=crop_black_borders,
                                                    crop_threshold=crop_threshold,
                                                    preprocessing_backend=preprocessing_backend,
                                                    loading_verbose=loading_verbose,
                                                    loading_pbar=loading_pbar)

    return DARTInferencePipeline(model_and_processing, device=_resolve_device(device))


class DARTInferencePipeline(nn.Module):
    def __init__(self, model_and_processing, device='cpu'):
        super().__init__()

        self.model = model_and_processing['model']
        self.preprocessing = model_and_processing['preprocessing']
        self.postprocessing = model_and_processing['postprocessing']
        self.cfg = model_and_processing['cfg']
        self.device = device
        self.model.to(device)

    def forward(self, x, as_dict=False):
        """Forward assumes a single image and returns the parameters"""
        with torch.inference_mode():
            x = self.preprocessing(x).unsqueeze(0).to(self.device)
            x = self.model(x)
            x = self.postprocessing(x).squeeze(0).cpu().numpy()
        if as_dict:
            return {t: x[i] for i, t in enumerate(self.cfg['retinal_traits'])}
        return x

    def forward_batch(self, x, as_dict=False):
        """Batch is assumed to come from a dataloader, so preprocessing has been applied to all inputs"""
        with torch.inference_mode():
            x = self.model(x)
            x = self.postprocessing(x).cpu().numpy()
        if as_dict:
            return {t: x[:, i] for i, t in enumerate(self.cfg['retinal_traits'])}
        return x

    def __repr__(self):
        out = f'{self.__class__.__name__}(model={self.cfg["model_name"]}, retinal traits={self.cfg["retinal_traits"]})\n'
        _out = self.preprocessing.__repr__().replace('\n', '\n\t')
        out += f'--- Preprocessing:\n\t{_out}\n'
        _out = self.model.__repr__().replace('\n', '\n\t')
        out += f'--- Model:\n\t{_out}\n'
        _out = self.postprocessing.__repr__().replace('\n', '\n\t')
        out += f'--- Postprocessing:\n\t{_out}\n'
        return out

    def update_device(self, device):
        self.device = device
        self.model.to(device)
