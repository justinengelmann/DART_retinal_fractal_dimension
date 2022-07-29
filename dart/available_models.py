from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

model_cfgs = {
    'resnet18': {
        'model_name': 'FD_resnet18',
        'url': 'https://github.com/justinengelmann/DART_retinal_fractal_dimension/releases/download/v0.1/resnet18_v0_1.pth',
        'local_path': '/home/jengelmann/DART_retinal_fractal_dimension/FD_resnet18_v0_1.pth',
        'input_size': (3, 224, 224),
        'norm_mean_stds': (IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        'output_size': 1,
        'zero_tensor_test_output': [-1.3943004608154297],
        'target_norm_method': 'meanstd',
        'target_norm_values': [(1.485399946403277, 0.0326135335465513)],
        'retinal_traits': ['FractalDimension_VAMPIRE_UKBioBank']
    }
}
