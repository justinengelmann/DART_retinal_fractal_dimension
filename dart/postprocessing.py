from torch import nn


def get_postprocessing(cfg: dict):
    # get 'target_norm_method' from cfg, default to 'none'
    target_norm_method = cfg.get('target_norm_method', 'none')

    if target_norm_method == 'none':
        return nn.Identity()
    elif target_norm_method == 'meanstd':
        # get 'target_norm_values' from cfg, fail if not present
        target_norm_values = cfg['target_norm_values']
        return DARTInvMeanStdScaler(mean=target_norm_values[0][0], std=target_norm_values[0][1])
    else:
        raise ValueError(f'Unknown target_norm_method {target_norm_method}')


class DARTInvMeanStdScaler(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std
        self.eps = 1e-9

    def forward(self, x):
        return x * (self.std + self.eps) + self.mean

    def __repr__(self):
        return f'{self.__class__.__name__}(mean={self.mean}, std={self.std})'
