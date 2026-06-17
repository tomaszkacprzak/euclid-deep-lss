# Copyright (C) 2022 ETH Zurich, Institute for Particle Physics and Astrophysics

"""PyTorch model wrapper lifecycle utilities."""

import shutil
import warnings
from pathlib import Path

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover - tensorboard is an optional runtime dependency
    SummaryWriter = None

try:
    from deepsphere import HealpyGCNN
except ImportError:  # pragma: no cover - optional DeepSphere dependency
    HealpyGCNN = None

from msfm.utils import logger

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("once", category=UserWarning)
LOGGER = logger.get_logger(__file__)


def _is_dist_initialized():
    return torch.distributed.is_available() and torch.distributed.is_initialized()


class BaseModel(object):
    """Base PyTorch model wrapper with training, checkpointing, summaries, and DDP support."""

    def __init__(
        self,
        network,
        input_shape=None,
        optimizer=None,
        optimizer_kwargs=None,
        summary_dir=None,
        checkpoint_dir=None,
        restore_checkpoint=False,
        max_checkpoints=3,
        init_step=0,
        distributed=False,
        device=None,
        scheduler=None,
        compile_model=False,
        summary_every=1,
        z_bank_size=None,
        # backward-compatible aliases
        strategy=None,
        xla=False,
        # DeepSphere
        n_side=None,
        indices=None,
        n_neighbors=20,
        max_batch_size=None,
        initial_Fin=None,
    ):
        optimizer_kwargs = optimizer_kwargs or {}
        if strategy is not None:
            LOGGER.warning("`strategy` is deprecated; use distributed=True with torch.distributed/DDP.")
            distributed = distributed or bool(strategy)
        if xla:
            LOGGER.warning("`xla` is deprecated and ignored by the PyTorch model wrapper; use compile_model=True.")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.distributed = distributed
        self.strategy = None
        self.xla = compile_model
        self.compile_model = compile_model
        self.summary_every = summary_every
        self.checkpoint_dir = checkpoint_dir
        self.max_checkpoints = max_checkpoints
        self.restore_from_checkpoint = restore_checkpoint
        self.global_step = int(init_step)
        self.epoch = 0
        self.scheduler = scheduler
        self.z_bank_size = z_bank_size
        self.z_bank = None
        self.z_bank_index = 0

        self.network = self._initialize_network(network, n_side, indices, n_neighbors, max_batch_size, initial_Fin)
        self.network.to(self.device)
        if input_shape is not None:
            self.build_network(input_shape)
        if compile_model:
            self.network = torch.compile(self.network)
        if distributed:
            if not _is_dist_initialized():
                raise RuntimeError("distributed=True requires torch.distributed.init_process_group() first")
            ddp_kwargs = {"device_ids": [self.device.index]} if self.device.type == "cuda" else {}
            self.network = DistributedDataParallel(self.network, **ddp_kwargs)

        self.optimizer = self._initialize_optimizer(optimizer, optimizer_kwargs)
        self.summary_dir = summary_dir
        if summary_dir is not None:
            if SummaryWriter is None:
                raise ImportError("torch.utils.tensorboard.SummaryWriter requires tensorboard to be installed")
            Path(summary_dir).mkdir(parents=True, exist_ok=True)
            self.summary_writer = SummaryWriter(summary_dir) if self.is_chief() else None
        else:
            self.summary_writer = None

        if checkpoint_dir is not None:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
            self.n_init_checkpoints = len(self._checkpoint_paths())
        else:
            self.n_init_checkpoints = 0

        if restore_checkpoint:
            self.restore_model()
        elif checkpoint_dir is not None and self.n_init_checkpoints:
            LOGGER.warning("Checkpoint directory is non-empty; save_model will refuse to overwrite from scratch.")
        else:
            LOGGER.info("The network is initialized from scratch.")

    def _initialize_network(self, network, n_side, indices, n_neighbors, max_batch_size, initial_Fin):
        if isinstance(network, list):
            if (n_side is None) and (indices is None):
                return nn.Sequential(*network)
            if HealpyGCNN is None:
                raise ImportError("DeepSphere is required for HealpyGCNN networks")
            return HealpyGCNN(
                nside=n_side, indices=indices, layers=network, n_neighbors=n_neighbors,
                max_batch_size=max_batch_size, initial_Fin=initial_Fin
            )
        if isinstance(network, nn.Module):
            return network
        raise ValueError(f"Invalid network {network}; expected torch.nn.Module or list of modules")

    def _initialize_optimizer(self, optimizer, optimizer_kwargs):
        if isinstance(optimizer, torch.optim.Optimizer):
            return optimizer
        if optimizer is None or optimizer == "adam":
            return torch.optim.Adam(self.parameters(), **optimizer_kwargs)
        if optimizer == "sgd":
            return torch.optim.SGD(self.parameters(), **optimizer_kwargs)
        raise NotImplementedError(f"Optimizer {optimizer} is not implemented")

    def module(self):
        return self.network.module if isinstance(self.network, DistributedDataParallel) else self.network

    def parameters(self):
        return self.module().parameters()

    def increment_step(self):
        self.global_step += 1

    def change_step(self, delta):
        self.global_step += int(delta)

    def set_step(self, step):
        self.global_step = int(step)

    def get_step(self):
        return int(self.global_step)

    def _checkpoint_paths(self):
        if self.checkpoint_dir is None:
            return []
        return sorted(Path(self.checkpoint_dir).glob("ckpt-*.pt"), key=lambda p: int(p.stem.split("-")[-1]))

    def save_model(self, config=None):
        if self.checkpoint_dir is None:
            raise ValueError("No checkpoint directory was declared during init; model cannot be saved.")
        if not self.restore_from_checkpoint and self.n_init_checkpoints != 0:
            raise Exception(f"Checkpoint directory {self.checkpoint_dir} was non-empty at initialization.")
        if not self.is_chief():
            return None
        path = Path(self.checkpoint_dir) / f"ckpt-{self.global_step}.pt"
        state = {
            "model": self.module().state_dict(),
            "optimizer": self.optimizer.state_dict() if self.optimizer is not None else None,
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "global_step": self.global_step,
            "epoch": self.epoch,
            "config": config or {},
        }
        torch.save(state, path)
        for old in self._checkpoint_paths()[:-self.max_checkpoints]:
            old.unlink(missing_ok=True)
        LOGGER.info(f"Successfully saved the model in {path}")
        return str(path)

    def restore_model(self):
        paths = self._checkpoint_paths()
        if not paths:
            raise ValueError(f"A non-empty checkpoint_dir {self.checkpoint_dir} has to be passed")
        return self.restore_model_from_checkpoint_path(str(paths[-1]))

    def restore_model_from_checkpoint_path(self, checkpoint_path, map_location=None, strict=True):
        state = torch.load(checkpoint_path, map_location=map_location or self.device)
        self.module().load_state_dict(state["model"], strict=strict)
        if state.get("optimizer") is not None and self.optimizer is not None:
            self.optimizer.load_state_dict(state["optimizer"])
        if state.get("scheduler") is not None and self.scheduler is not None:
            self.scheduler.load_state_dict(state["scheduler"])
        self.global_step = int(state.get("global_step", self.global_step))
        self.epoch = int(state.get("epoch", self.epoch))
        LOGGER.info(f"Network successfully restored from checkpoint {checkpoint_path}.")
        return state

    def build_network(self, input_shape):
        if hasattr(self.network, "build"):
            self.network.build(input_shape=input_shape)
        else:
            with torch.no_grad():
                dummy = torch.zeros(input_shape, device=self.device)
                self.network(dummy)

    def print_summary(self, **kwargs):
        LOGGER.info(str(self.network))

    def write_summary(self, label, value, summary_type="scalar", skip=False):
        if self.summary_writer is None or skip or not self.is_chief():
            return
        if self.global_step % self.summary_every != 0:
            return
        if torch.is_tensor(value):
            value = value.detach().cpu()
        if summary_type == "scalar":
            self.summary_writer.add_scalar(label, value, self.global_step)
        elif summary_type == "histogram":
            self.summary_writer.add_histogram(label, value, self.global_step)
        elif summary_type == "image":
            self.summary_writer.add_image(label, value, self.global_step)
        else:
            raise ValueError(f"Invalid summary type {summary_type} was passed")

    def create_temp_dir(self, chief_dir):
        temp_dir = Path(chief_dir) / f"temp_worker_{torch.distributed.get_rank() if _is_dist_initialized() else 0}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        return str(temp_dir)

    def copy_chief_to_temp_dir(self, chief_dir, temp_dir):
        Path(temp_dir).mkdir(parents=True, exist_ok=True)
        for path in Path(chief_dir).iterdir():
            if path.is_file():
                shutil.copy2(path, Path(temp_dir) / path.name)

    def delete_temp_summaries(self):
        if self.summary_dir is not None and not self.is_chief():
            shutil.rmtree(self.summary_dir, ignore_errors=True)

    def is_chief(self):
        return (not _is_dist_initialized()) or torch.distributed.get_rank() == 0

    def _all_gather_tensor(self, tensor):
        if not _is_dist_initialized() or torch.distributed.get_world_size() == 1:
            return tensor
        gathered = [torch.empty_like(tensor) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(gathered, tensor)
        return torch.cat(gathered, dim=0)

    def _compute_vicreg_var_cov_loss(self, z):
        batch_size = z.shape[0]
        feature_dim = z.shape[1]
        z_centered = z - torch.mean(z, dim=0, keepdim=True)
        std = torch.sqrt(torch.mean(z_centered.square(), dim=0) + 1e-4)
        var_loss = torch.mean((std - 1.0).square())
        cov_matrix = torch.matmul(z_centered.T, z_centered) / max(batch_size - 1, 1)
        cov_loss = torch.sum(cov_matrix.square()) - torch.sum(torch.linalg.diagonal(cov_matrix).square())
        cov_loss = cov_loss / max(feature_dim**2 - feature_dim, 1)
        return var_loss, cov_loss

    def _compute_vicreg_invariance_loss(self, z, pair_ids):
        match = torch.all(pair_ids[:, None, :] == pair_ids[None, :, :], dim=-1)
        mask = match & ~torch.eye(z.shape[0], dtype=torch.bool, device=z.device)
        mask_f = mask.to(z.dtype)
        n_pair_entries = torch.sum(mask_f)
        self.write_summary("z_invariance/n_positive_pairs", n_pair_entries / 2.0)
        self.write_summary("z_invariance/n_anchored_samples", torch.sum(torch.any(mask, dim=1).to(z.dtype)))
        diff = z[:, None, :] - z[None, :, :]
        pairwise_mse = torch.mean(diff.square(), dim=-1)
        return torch.where(n_pair_entries > 0, torch.sum(pairwise_mse * mask_f) / n_pair_entries, torch.zeros((), device=z.device, dtype=z.dtype))

    def _compute_mmd_loss(self, z, interpretable=False):
        batch_size, feature_dim = z.shape
        z_gaussian = torch.randn_like(z)
        dim_scale = torch.sqrt(torch.tensor(feature_dim, device=z.device, dtype=z.dtype))
        def rbf_kernel(x, y):
            xx = torch.sum(x.square(), dim=1, keepdim=True)
            yy = torch.sum(y.square(), dim=1, keepdim=True)
            distances = torch.relu(xx - 2 * torch.matmul(x, y.T) + yy.T)
            kernel_matrix = torch.zeros_like(distances)
            for bandwidth in (0.1 * dim_scale, 1.0 * dim_scale, 10.0 * dim_scale):
                kernel_matrix = kernel_matrix + torch.exp(-distances / (2 * bandwidth**2))
            return kernel_matrix / 3
        k_zz = rbf_kernel(z, z)
        k_zg = rbf_kernel(z, z_gaussian)
        mmd_loss = torch.sum(k_zz) / (batch_size * batch_size) - 2 * torch.sum(k_zg) / (batch_size * batch_size)
        if interpretable:
            mmd_loss = mmd_loss + torch.sum(rbf_kernel(z_gaussian, z_gaussian)) / (batch_size * batch_size)
        return mmd_loss

    def _compute_sw_loss(self, z, num_projections=None, method="analytical"):
        batch_size, feature_dim = z.shape
        if num_projections is None:
            num_projections = max(512, feature_dim)
        projections = torch.randn((feature_dim, num_projections), device=z.device, dtype=z.dtype)
        projections = torch.nn.functional.normalize(projections, p=2, dim=0)
        sorted_z = torch.sort(torch.matmul(z, projections), dim=0).values
        if method == "analytical":
            probs = (torch.arange(batch_size, device=z.device, dtype=z.dtype) + 0.5) / batch_size
            sorted_gaussian = torch.distributions.Normal(0, 1).icdf(probs).unsqueeze(-1)
        elif method == "sample":
            sorted_gaussian = torch.sort(torch.matmul(torch.randn_like(z), projections), dim=0).values
        else:
            raise ValueError(f"Invalid method {method}. Must be 'sample' or 'analytical'.")
        return torch.mean((sorted_z - sorted_gaussian).square())

    def _update_and_get_z_bank(self, z):
        z = self._all_gather_tensor(z)
        world_size = torch.distributed.get_world_size() if _is_dist_initialized() else 1
        if self.z_bank_size is None:
            return z, float(world_size)
        if self.z_bank is None:
            LOGGER.info(f"Initializing z memory bank with size {self.z_bank_size}")
            self.z_bank = torch.randn((self.z_bank_size, z.shape[-1]), device=z.device, dtype=z.dtype)
            self.z_bank_index = 0
        batch_size = z.shape[0]
        indices = (torch.arange(batch_size, device=z.device) + self.z_bank_index) % self.z_bank_size
        self.z_bank[indices] = z.detach()
        self.z_bank_index = int((self.z_bank_index + batch_size) % self.z_bank_size)
        z_loss_input = torch.cat([z, self.z_bank], dim=0)
        z_scale = (batch_size + self.z_bank_size) / batch_size
        self.write_summary("z_bank/scale", z_scale)
        return z_loss_input, z_scale * world_size

    def train_step(self, *args, **kwargs):
        return self.base_train_step(*args, **kwargs)

    def base_train_step(self, input_tensor, loss_function, input_labels=None, clip_by_value=None, clip_by_norm=None, clip_by_global_norm=None, l2_norm_weight=None, z_weight=None, z_type=None, z_layer="last", pair_ids=None):
        self.network.train()
        input_tensor = input_tensor.to(self.device) if torch.is_tensor(input_tensor) else input_tensor
        input_labels = input_labels.to(self.device) if torch.is_tensor(input_labels) else input_labels
        self.optimizer.zero_grad(set_to_none=True)
        predictions = self.network(input_tensor)
        loss = loss_function(predictions) if input_labels is None else loss_function(predictions, input_labels)
        self.write_summary("loss/main", loss)
        params = [p for p in self.parameters() if p.requires_grad]
        if l2_norm_weight is not None:
            l2_loss = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(p) for p in params]))
            self.write_summary("loss/l2_reg", l2_loss)
            loss = loss + l2_norm_weight * l2_loss
        if z_weight is not None:
            if z_layer == "penultimate":
                if not hasattr(self.module(), "children"):
                    raise ValueError("penultimate z_layer requires an iterable module")
                z_features = input_tensor
                for layer in list(self.module().children())[:-1]:
                    z_features = layer(z_features)
            elif z_layer == "last":
                z_features = predictions
            else:
                raise ValueError(f"Invalid z_layer '{z_layer}', must be 'penultimate' or 'last'")
            if z_type == "vicreg":
                var_w, cov_w, inv_w = z_weight.get("variance"), z_weight.get("covariance"), z_weight.get("invariance")
                if var_w is not None or cov_w is not None:
                    z_input, z_scale = self._update_and_get_z_bank(z_features)
                    var_loss, cov_loss = self._compute_vicreg_var_cov_loss(z_input)
                    if var_w is not None:
                        loss = loss + var_w * z_scale * var_loss
                    if cov_w is not None:
                        loss = loss + cov_w * z_scale * cov_loss
                if inv_w is not None:
                    if pair_ids is None:
                        raise AssertionError("pair_ids must be passed when z_weight['invariance'] is set")
                    z_inv_input = self._all_gather_tensor(z_features)
                    pair_ids_stacked = torch.stack([self._all_gather_tensor(p.to(self.device)) for p in pair_ids], dim=-1)
                    loss = loss + inv_w * (_is_dist_initialized() and torch.distributed.get_world_size() or 1) * self._compute_vicreg_invariance_loss(z_inv_input, pair_ids_stacked)
            elif z_type == "mmd":
                z_input, z_scale = self._update_and_get_z_bank(z_features)
                loss = loss + z_weight * z_scale * self._compute_mmd_loss(z_input)
            elif z_type == "sw":
                z_input, z_scale = self._update_and_get_z_bank(z_features)
                loss = loss + z_weight * z_scale * self._compute_sw_loss(z_input)
            else:
                raise ValueError(f"Invalid z_type {z_type}. Must be 'vicreg', 'mmd', or 'sw'.")
        self.write_summary("loss/total", loss)
        loss.backward()
        with torch.no_grad():
            for p in params:
                if p.grad is not None:
                    p.grad = torch.where(torch.isfinite(p.grad), p.grad, torch.zeros_like(p.grad))
        if clip_by_value is not None:
            torch.nn.utils.clip_grad_value_(params, clip_by_value if not isinstance(clip_by_value, (tuple, list)) else max(abs(clip_by_value[0]), abs(clip_by_value[1])))
        if clip_by_norm is not None:
            torch.nn.utils.clip_grad_norm_(params, clip_by_norm)
        grad_norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(p.grad) for p in params if p.grad is not None]))
        self.write_summary("global_grad_norm", grad_norm)
        if clip_by_global_norm is not None:
            torch.nn.utils.clip_grad_norm_(params, clip_by_global_norm)
            post = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(p.grad) for p in params if p.grad is not None]))
            self.write_summary("global_grad_norm_post_clip", post)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.write_summary("params/global_norm", torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(p.detach()) for p in params])))
        lr = self.optimizer.param_groups[0].get("lr")
        self.write_summary("learning_rate", lr)
        self.increment_step()
        return loss.detach()

    def distributed_train_step(self, *args, **kwargs):
        if not isinstance(self.network, DistributedDataParallel):
            raise RuntimeError("distributed_train_step requires a DistributedDataParallel-wrapped network")
        return self.base_train_step(*args, **kwargs)

    def __call__(self, input_tensor, training=False, numpy=False, layer=None, *args, **kwargs):
        self.network.train(training)
        if torch.is_tensor(input_tensor):
            input_tensor = input_tensor.to(self.device)
        with torch.set_grad_enabled(training):
            if layer is None:
                preds = self.network(input_tensor, *args, **kwargs)
            else:
                preds = input_tensor
                for child in list(self.module().children())[:layer]:
                    preds = child(preds)
        return preds.detach().cpu().numpy() if numpy else preds

    def torch_call(self, input_tensor, training=False, *args, **kwargs):
        return self.__call__(input_tensor, training=training, *args, **kwargs)
