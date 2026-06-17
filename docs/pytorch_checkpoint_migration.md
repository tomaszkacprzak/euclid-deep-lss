# PyTorch checkpoint migration note

The model wrappers in `deep_lss.models.base_model`, `deep_lss.models.grid_model`, and
`deep_lss.models.delta_model` now use PyTorch modules and checkpoint files written with
`torch.save`.

New checkpoints are dictionaries with these keys:

- `model`: `torch.nn.Module.state_dict()`
- `optimizer`: `torch.optim.Optimizer.state_dict()` when an optimizer is configured
- `scheduler`: scheduler state when a scheduler is configured
- `global_step`: integer training step
- `epoch`: integer epoch counter
- `config`: user-provided metadata

Existing TensorFlow `.ckpt` files do **not** load directly into these PyTorch modules.
To migrate an old run, export the TensorFlow model weights to an intermediate format
(such as NumPy arrays or HDF5), create an equivalent PyTorch module with matching layer
names and tensor layouts, transpose kernels where required by the framework convention,
and then build a PyTorch `state_dict` that can be loaded with `load_state_dict`.
Optimizer slot variables from TensorFlow should be treated as incompatible; restart the
PyTorch optimizer unless you have a custom one-off conversion script for that optimizer.
