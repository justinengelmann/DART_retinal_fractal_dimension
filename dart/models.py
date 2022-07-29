import logging

import timm
import torch
from timm.models.layers.adaptive_avgmax_pool import SelectAdaptivePool2d
from timm.models.layers.attention_pool2d import AttentionPool2d
from timm.models.layers.create_act import get_act_layer as timm_get_act_layer
from torch import nn
from torch.nn import functional as F


class TimmRegressionModel(nn.Module):
    """Simple regression model using timm's models. Optionally uses a custom MLP as head."""

    def __init__(self,
                 model_name='resnet18',
                 pretrained=True,
                 num_outputs=1,
                 input_size=(512, 512),
                 num_channels=3,
                 use_custom_head=False,
                 custom_head_kwargs=None):
        super().__init__()
        self.model_name = model_name
        self.pretrained = pretrained
        self.num_outputs = num_outputs
        self.input_size = input_size
        self.num_channels = num_channels
        self.use_custom_head = use_custom_head
        self.custom_head_kwargs = custom_head_kwargs or {}

        try:
            self.base_model = timm.create_model(model_name,
                                                img_size=input_size,
                                                pretrained=pretrained, num_classes=num_outputs, in_chans=num_channels)
        except TypeError:
            self.base_model = timm.create_model(model_name,
                                                pretrained=pretrained, num_classes=num_outputs, in_chans=num_channels)

        if self.use_custom_head:
            base_model_output_shape = self._get_base_model_output_shape()

            # set in_channels and input_size if not specified
            if 'in_channels' not in self.custom_head_kwargs:
                self.custom_head_kwargs['in_channels'] = base_model_output_shape[1]
            if 'input_size' not in self.custom_head_kwargs:
                self.custom_head_kwargs['input_size'] = base_model_output_shape[2:]

            self.head = CustomHead(**self.custom_head_kwargs)
        else:
            if self.custom_head_kwargs is not None:
                logging.warning('custom_head_kwargs is not used when use_custom_head is False')
            self.head = None

        self._validate_output_shape()

        # TODO fix
        self.base_model_param_size = sum(p.numel() for p in self.parameters())
        self.head_param_size = sum(p.numel() for p in self.head.parameters()) if self.head is not None else 0
        self.total_param_size = self.base_model_param_size + self.head_param_size

    def forward(self, x):
        if self.use_custom_head:
            x = self.base_model.forward_features(x)
            return self.head(x)
        else:
            return self.base_model(x)

    def _get_base_model_output_shape(self):
        return self.base_model.forward_features(torch.zeros(8, self.num_channels, *self.input_size)).shape

    def _validate_output_shape(self):
        output_shape = self.forward(torch.zeros([8, self.num_channels, *self.input_size])).shape
        expected_output_shape = torch.Size([8, self.num_outputs])
        assert output_shape == expected_output_shape, \
            f'actual shape {output_shape} != expected shape {expected_output_shape}, (8 is batchdim)'

    def __repr__(self):
        return f'{self.__class__.__name__}(model_name={self.model_name}, num_outputs={self.num_outputs}, input_size={self.input_size}, num_channels={self.num_channels},\n' \
               f'use_custom_head={self.use_custom_head}, custom_head_kwargs={self.custom_head_kwargs})\n' \
               f'(basemodel): {type(self.base_model).__name__} ({self.base_model_param_size // 1e6:.4f}M parameters)\n' \
               f'(head): {self.head if self.use_custom_head else "None"} ({self.head_param_size // 1e6:.4f}M parameters)\n' \
               f'Total param count: {self.total_param_size // 1e6:.4f}M parameters'


class CustomHead(nn.Module):
    """Custom head for regression model."""

    def __init__(self,
                 in_channels,
                 num_outputs=1,
                 input_size=None,
                 pool_type='avg',
                 output_act='none',
                 hidden_layer_sizes=(64,),
                 hidden_act='gelu',
                 input_drop_rate=0.,
                 hidden_drop_rate=0.,
                 norm='layer'
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.num_outputs = num_outputs
        self.input_size = input_size
        self.output_act = output_act
        self.hidden_layer_sizes = hidden_layer_sizes
        self.hidden_act = hidden_act
        self.input_drop_rate = input_drop_rate
        self.hidden_drop_rate = hidden_drop_rate
        self.norm = norm
        self.pool_type = pool_type

        attn_out_features = hidden_layer_sizes[0] if hidden_layer_sizes is not None else num_outputs
        self.global_pool, num_pooled_features = self._create_pool(num_features=in_channels,
                                                                  pool_type=pool_type,
                                                                  input_size=input_size,
                                                                  attn_out_features=attn_out_features)
        self.num_pooled_features = num_pooled_features
        self.mlp = self._create_mlp(num_pooled_features=num_pooled_features,
                                    num_outputs=num_outputs,
                                    hidden_layer_sizes=hidden_layer_sizes,
                                    hidden_act=hidden_act,
                                    output_act=output_act,
                                    hidden_drop_rate=hidden_drop_rate,
                                    norm=norm)
        self.output_act = get_act_layer(output_act)()

    def forward(self, x):
        x = self.global_pool(x)
        if self.input_drop_rate > 0:
            x = F.dropout(x, p=self.input_drop_rate, training=self.training)
        x = self.mlp(x)
        x = self.output_act(x)
        return x

    @staticmethod
    def _create_pool(num_features, pool_type, input_size, attn_out_features=None):
        if len(input_size) == 0:
            # already flattened, or doesn't need flattening (e.g. ViT)
            return nn.Identity(), num_features

        if pool_type != 'attention':
            pool = SelectAdaptivePool2d(pool_type=pool_type, flatten=True)
            num_pooled_features = num_features * pool.feat_mult()
            return pool, num_pooled_features
        else:
            pool = AttentionPool2d(in_features=num_features, feat_size=input_size, out_features=attn_out_features,
                                   embed_dim=32, num_heads=1, qkv_bias=True)
            num_pooled_features = attn_out_features or num_features
            return pool, num_pooled_features

    def _create_mlp(self, num_pooled_features, num_outputs, output_act,
                    hidden_layer_sizes, hidden_act, hidden_drop_rate, norm):
        if hidden_layer_sizes is None:
            return nn.Linear(num_pooled_features, num_outputs)

        mlp_layers = [
            self._create_layer(num_pooled_features, hidden_layer_sizes[0], hidden_act, hidden_drop_rate, norm)]
        for idx in range(1, len(hidden_layer_sizes)):
            mlp_layers.append(self._create_layer(hidden_layer_sizes[idx - 1], hidden_layer_sizes[idx], hidden_act,
                                                 hidden_drop_rate, norm))
        mlp_layers.append(nn.Linear(hidden_layer_sizes[-1], num_outputs))
        return nn.Sequential(*mlp_layers)

    @staticmethod
    def _create_layer(in_features, out_features, hidden_act, hidden_drop_rate, norm):
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            get_1dnorm_layer(norm, out_features),
            nn.Dropout(p=hidden_drop_rate),
            get_act_layer(hidden_act)()
        )

    def __repr__(self):
        return f'{self.__class__.__name__}(in_channels={self.in_channels}, num_outputs={self.num_outputs}, input_size={self.input_size}, pool_type={self.pool_type}, output_act={self.output_act},' \
               f' hidden_layer_sizes={self.hidden_layer_sizes}, hidden_act={self.hidden_act}, input_drop_rate={self.input_drop_rate}, hidden_drop_rate={self.hidden_drop_rate}, norm={self.norm})'


def get_act_layer(activation):
    if activation in ['identity', 'none'] or activation is None:
        return nn.Identity
    else:
        return timm_get_act_layer(activation)


def get_1dnorm_layer(norm, num_features):
    # select nn class for specified norm (bn, instance, layer, etc)
    if norm in ['bn', 'batch', 'batchnorm', 'batch_norm']:
        return nn.BatchNorm1d(num_features=num_features)
    elif norm in ['in', 'instance', 'instancenorm', 'instance_norm']:
        return nn.InstanceNorm1d(num_features=num_features)
    elif norm in ['ln', 'layer', 'layernorm', 'layer_norm']:
        return nn.LayerNorm(normalized_shape=(num_features,))
    elif norm in ['none', 'identity'] or norm is None:
        return nn.Identity()
    else:
        raise ValueError(
            f'Unknown norm {norm}. Specify one of batch, instance, layer, none')
