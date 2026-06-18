"""Static guard for direct TensorFlow imports.

TensorFlow is still used by the legacy model stack, but direct TFRecord input
plumbing should be centralized in deep_lss/data/tfrecords.py.  Modules listed
in ALLOWED_TENSORFLOW_IMPORTS are intentional TensorFlow model/distribution
helpers and may be migrated separately.
"""

from __future__ import annotations

import ast
from pathlib import Path

ALLOWED_TENSORFLOW_IMPORTS = {
    Path("deep_lss/data/tfrecords.py"),
    Path("deep_lss/apps/run_cls_training+evaluation.py"),
    Path("deep_lss/compat/deprecated/estimators.py"),
    Path("deep_lss/compat/deprecated/layers.py"),
    Path("deep_lss/models/base_model.py"),
    Path("deep_lss/models/delta_model.py"),
    Path("deep_lss/models/grid_model.py"),
    Path("deep_lss/nets/custom_layers.py"),
    Path("deep_lss/nets/gaussian_mixture.py"),
    Path("deep_lss/compat/legacy_nets/bary_net_partial.py"),
    Path("deep_lss/compat/legacy_nets/bary_net_partial_v2.py"),
    Path("deep_lss/compat/legacy_nets/bary_net_partial_v3.py"),
    Path("deep_lss/compat/legacy_nets/big_resnet.py"),
    Path("deep_lss/compat/legacy_nets/big_resnet_partial.py"),
    Path("deep_lss/compat/legacy_nets/small_resnet.py"),
    Path("deep_lss/compat/legacy_nets/small_resnet_partial.py"),
    Path("deep_lss/compat/legacy_nets/small_resnet_partial_v2.py"),
    Path("deep_lss/compat/legacy_nets/small_resnet_partial_v3.py"),
    Path("deep_lss/compat/legacy_nets/small_resnet_v2.py"),
    Path("deep_lss/nets/maps_plus_cls_network.py"),
    Path("deep_lss/nets/mlp.py"),
    Path("deep_lss/nets/normalizing_flow.py"),
    Path("deep_lss/nets/one_d_conv.py"),
    Path("deep_lss/nets/regression_head.py"),
    Path("deep_lss/nets/resnet.py"),
    Path("deep_lss/nets/transformer.py"),
    Path("deep_lss/utils/cls_evaluation.py"),
    Path("deep_lss/utils/configuration.py"),
    Path("deep_lss/utils/delta_loss.py"),
    Path("deep_lss/utils/distribute/general.py"),
    Path("deep_lss/utils/distribute/horovod.py"),
    Path("deep_lss/utils/distribute/tensorflow.py"),
    Path("deep_lss/utils/evaluation.py"),
    Path("deep_lss/utils/likelihood_loss.py"),
    Path("deep_lss/utils/mutual_info_loss.py"),
    Path("deep_lss/utils/optimization.py"),
    Path("deep_lss/utils/summary.py"),
}


def _imports_tensorflow(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name == "tensorflow" or alias.name.startswith("tensorflow.") for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            if node.module and (node.module == "tensorflow" or node.module.startswith("tensorflow.")):
                return True
    return False


def test_direct_tensorflow_imports_are_guarded():
    root = Path(__file__).resolve().parents[1]
    offenders = []
    for path in root.joinpath("deep_lss").rglob("*.py"):
        rel = path.relative_to(root)
        if rel in ALLOWED_TENSORFLOW_IMPORTS:
            continue
        if _imports_tensorflow(ast.parse(path.read_text(), filename=str(rel))):
            offenders.append(str(rel))
    assert offenders == []
