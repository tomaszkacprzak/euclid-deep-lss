"""PyTorch transformer-based network builders."""

from __future__ import annotations

from deep_lss.nets import deepsphere_torch as dst
from deep_lss.nets.regression_head import get_regression_head


class ViTLayers:
    def __init__(
        self,
        out_features=6,
        base_channels=None,
        downsampling_layers=None,
        hidden_dim=128,
        healpix_patch_fac=4,
        attention_heads=4,
        transformer_layers=4,
        pos_encoding=True,
        layer_norm=True,
        second_to_last_features=None,
        dropout_rate=None,
        activation="relu",
        smoothing_kwargs=None,
    ) -> None:
        self.layers = []
        if smoothing_kwargs is not None:
            self.layers.append(dst.healpy_smoothing(**smoothing_kwargs))
        if base_channels is not None and downsampling_layers is not None:
            n_channels = base_channels
            for _ in range(downsampling_layers):
                self.layers.append(dst.healpy_pseudo_conv(p=1, Fout=n_channels, activation=activation))
                n_channels *= 2
        self.layers.append(
            dst.healpy_vit(
                p=healpix_patch_fac,
                key_dim=hidden_dim,
                num_heads=attention_heads,
                positional_encoding=pos_encoding,
                n_layers=transformer_layers,
                layer_norm=layer_norm,
                activation=activation,
            )
        )
        self.layers.extend(
            get_regression_head(
                out_features=out_features,
                head_type="dense",
                second_to_last_features=second_to_last_features,
                activation=activation,
                dropout_rate=dropout_rate,
            )
        )

    def get_layers(self):
        return self.layers


class GTLayers:
    def __init__(
        self,
        out_features,
        base_channels=32,
        downsampling_layers=4,
        hidden_dim=128,
        attention_heads=4,
        transformer_layers=4,
        pos_encoding=True,
        layer_norm=True,
        second_to_last_features=None,
        dropout_rate=None,
        activation="relu",
        smoothing_kwargs=None,
    ) -> None:
        self.layers = []
        if smoothing_kwargs is not None:
            self.layers.append(dst.healpy_smoothing(**smoothing_kwargs))
        n_channels = base_channels
        for _ in range(downsampling_layers):
            self.layers.append(dst.healpy_pseudo_conv(p=1, Fout=n_channels, activation=activation))
            n_channels *= 2
        self.layers.append(
            dst.healpy_transformer(
                key_dim=hidden_dim,
                num_heads=attention_heads,
                positional_encoding=pos_encoding,
                n_layers=transformer_layers,
                layer_norm=layer_norm,
                activation=activation,
            )
        )
        self.layers.extend(
            get_regression_head(
                out_features=out_features,
                head_type="dense",
                second_to_last_features=second_to_last_features,
                activation=activation,
                dropout_rate=dropout_rate,
            )
        )

    def get_layers(self):
        return self.layers
