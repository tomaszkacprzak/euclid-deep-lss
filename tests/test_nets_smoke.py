"""Module-level PyTorch smoke tests for deep_lss.nets."""

import pytest

torch = pytest.importorskip("torch")


def test_custom_layers_smoke():
    from deep_lss.nets.custom_layers import MeanBinningLayer, PowerSpectrumSmoothingLayer

    x = torch.arange(2 * 6 * 3, dtype=torch.float32).reshape(2, 6, 3)
    assert MeanBinningLayer([0, 2, 4, 6])(x).shape == (2, 3, 3)
    assert PowerSpectrumSmoothingLayer(6)(torch.ones(2, 6)).shape == (2, 6)


def test_mlp_smoke():
    from deep_lss.nets.mlp import MultiLayerPerceptron

    model = MultiLayerPerceptron(output_size=3, num_hidden_units=8, num_layers=2, dropout_rate=0.1)
    assert model(torch.randn(4, 5)).shape == (4, 3)


def test_gaussian_mixture_smoke():
    from deep_lss.nets.gaussian_mixture import GaussianMixtureModel

    model = GaussianMixtureModel(dim_theta=2, dim_summary=5, num_components=3, full_covariance=False)
    theta, summary = torch.randn(4, 2), torch.randn(4, 5)
    assert model.log_prob(theta, summary).shape == (4,)
    assert model.mean(summary).shape == (4, 2)


def test_normalizing_flow_smoke():
    from deep_lss.nets.normalizing_flow import NormalizingFlowModel

    model = NormalizingFlowModel(dim_theta=4, dim_summary=5, num_layers=2)
    theta, summary = torch.randn(4, 4), torch.randn(4, 5)
    assert model.log_prob(theta, summary).shape == (4,)
    assert model.inverse(theta, summary).shape == (4, 4)
    assert model.mean(summary, n_samples=3).shape == (4, 4)


def test_regression_head_smoke():
    from deep_lss.nets.regression_head import get_cls_embedding_layers, get_regression_head

    x = torch.randn(4, 3, 2)
    for layer in get_regression_head(out_features=2, dense_layers=[4]):
        x = layer(x)
    assert x.shape == (4, 2)

    y = torch.randn(4, 5)
    for layer in get_cls_embedding_layers([7, 6]):
        y = layer(y)
    assert y.shape == (4, 6)


def test_one_d_residual_block_smoke():
    from deep_lss.nets.one_d_conv import OneDResidualBlock

    block = OneDResidualBlock(filters=3, kernel_size=3)
    assert block(torch.randn(4, 8, 3)).shape == (4, 8, 3)


def test_maps_plus_cls_network_smoke(monkeypatch):
    from deep_lss.nets.maps_plus_cls_network import MapsPlusCLSNetwork
    from deep_lss.nets.regression_head import get_cls_embedding_layers, get_regression_head

    monkeypatch.setattr("deep_lss.nets.maps_plus_cls_network.MapGCNN", lambda **kwargs: torch.nn.Identity())
    model = MapsPlusCLSNetwork(
        conv_layers=[],
        cls_embedding_layers=get_cls_embedding_layers([4]),
        regression_head_layers=get_regression_head(out_features=2, dense_layers=[5])[1:],
        n_side=1,
        tfr_n_side=2,
        indices=None,
        n_neighbors=1,
        max_batch_size=4,
        initial_Fin=3,
        n_cls_bins=2,
        l_min_per_pair=[0, 0],
        l_max_per_pair=[6, 6],
    )
    assert model((torch.randn(4, 5, 3), torch.randn(4, 6, 2))).shape == (4, 2)
