"""Network registry with lazy imports.

Importing :mod:`deep_lss.nets.deepsphere_torch` must not require TensorFlow-only
legacy modules, so network classes are resolved only when requested.
"""

_NETWORK_IMPORTS = {
    "resnet": ("deep_lss.nets.resnet", "ResNetLayers"),
    "vision_transformer": ("deep_lss.nets.transformer", "ViTLayers"),
    "graph_transformer": ("deep_lss.nets.transformer", "GTLayers"),
    "one_d_conv": ("deep_lss.nets.one_d_conv", "OneDConvLayers"),
    "maps_plus_cls": ("deep_lss.nets.maps_plus_cls_network", "MapsPlusCLSNetwork"),
}


class _LazyNetworkRegistry(dict):
    def __getitem__(self, key):
        module_name, attr_name = _NETWORK_IMPORTS[key]
        import importlib

        return getattr(importlib.import_module(module_name), attr_name)

    def get(self, key, default=None):
        return self[key] if key in self else default


NETWORKS = _LazyNetworkRegistry({name: None for name in _NETWORK_IMPORTS if name != "maps_plus_cls"})


def __getattr__(name):
    mapping = {
        "ResNetLayers": ("deep_lss.nets.resnet", "ResNetLayers"),
        "ViTLayers": ("deep_lss.nets.transformer", "ViTLayers"),
        "GTLayers": ("deep_lss.nets.transformer", "GTLayers"),
        "OneDConvLayers": ("deep_lss.nets.one_d_conv", "OneDConvLayers"),
        "MapsPlusCLSNetwork": ("deep_lss.nets.maps_plus_cls_network", "MapsPlusCLSNetwork"),
    }
    if name not in mapping:
        raise AttributeError(name)
    import importlib

    module_name, attr_name = mapping[name]
    return getattr(importlib.import_module(module_name), attr_name)


__all__ = ["NETWORKS", "ResNetLayers", "ViTLayers", "GTLayers", "OneDConvLayers", "MapsPlusCLSNetwork"]
