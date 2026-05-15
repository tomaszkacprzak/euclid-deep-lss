import argparse, h5py, os, yaml
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

for gpu in tf.config.list_physical_devices(device_type="GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

from tqdm import tqdm

from msfm.utils import files

from deep_lss.models.grid_model import GridLossModel
from deep_lss.nets.mlp import MultiLayerPerceptron
from deep_lss.utils import evaluation
from deep_lss.utils.mutual_info_loss import distance_correlation

from msi.utils import dataset, preprocessing


class EarlyStopper:
    def __init__(self, patience, min_delta=0.0, min_steps=0):
        self.patience = patience
        self.min_delta = min_delta
        self.min_steps = min_steps
        self.best_loss = float("inf")
        self.wait = 0

    def update(self, loss):
        """Returns True if this is a new best (caller should save checkpoint)."""
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.wait = 0
            return True
        self.wait += 1
        return False

    def should_stop(self, step):
        return step >= self.min_steps and self.wait >= self.patience


def setup():
    parser = argparse.ArgumentParser(
        description="Train an MLP summary network on binned power spectra (Cls) using the mutual information loss."
    )
    parser.add_argument("--msfm_config", required=True)
    parser.add_argument("--dlss_config", required=True)
    parser.add_argument("--mlp_config", required=True)
    parser.add_argument("--vmim_config", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--model_name", default="model")
    parser.add_argument("--restore_checkpoint", action="store_true")

    # Observation inclusion flags (all default off)
    parser.add_argument("--include_grid", action="store_true")
    parser.add_argument("--n_grid_examples", type=int, default=16)
    parser.add_argument("--include_des", action="store_true")
    parser.add_argument("--include_bench", action="store_true")

    return parser.parse_args()


def main():
    args = setup()

    msfm_conf = files.load_config(args.msfm_config)
    from msfm.utils import input_output

    dlss_conf = input_output.read_yaml(args.dlss_config)
    mlp_conf = input_output.read_yaml(args.mlp_config)
    vmim_conf = input_output.read_yaml(args.vmim_config)

    common = dlss_conf["dset"]["common"]
    with_lensing = common["with_lensing"]
    with_clustering = common["with_clustering"]
    with_cross_z = common["with_cross_z"]
    with_cross_probe = common["with_cross_probe"]
    ggl_only = common.get("ggl_only", False)

    if with_lensing and not with_clustering:
        probe = "lensing"
    elif with_clustering and not with_lensing:
        probe = "clustering"
    elif with_lensing and with_clustering:
        probe = "combined"
    else:
        probe = "cross"

    params = dlss_conf["dset"]["training"]["params"]
    n_params = len(params)
    n_summary = 2 * n_params

    n_steps = mlp_conf["n_steps"]
    batch_size = mlp_conf["batch_size"]
    log_every = mlp_conf["log_every"]
    vali_every = mlp_conf["vali_every"]

    pred_dir = os.path.join(args.out_dir, args.model_name)
    os.makedirs(pred_dir, exist_ok=True)

    pred_file = os.path.join(pred_dir, f"preds_{n_steps}.h5")

    print(f"probe       = {probe}")
    print(f"pred_dir    = {pred_dir}")
    print(f"pred_file   = {pred_file}")
    print(f"params      = {params}")
    print(f"n_steps     = {n_steps}")

    with open(os.path.join(pred_dir, "configs.yaml"), "w") as f:
        yaml.dump_all([mlp_conf, vmim_conf, dlss_conf, msfm_conf], f)

    cl_dset_train, cl_dset_test, out_dict = dataset.get_binned_power_spectra_dset(
        args.data_dir,
        msfm_conf=msfm_conf,
        dlss_conf=dlss_conf,
        params=params,
        with_lensing=with_lensing,
        with_clustering=with_clustering,
        with_cross_z=with_cross_z,
        with_cross_probe=with_cross_probe,
        ggl_only=ggl_only,
        batch_size=batch_size,
        apply_log=True,
        standardize=False,
    )

    n_cls = out_dict["grid/cls/train"].shape[-1]

    summary_net = MultiLayerPerceptron(
        output_size=n_summary,
        num_hidden_units=mlp_conf["num_hidden_units"],
        num_layers=mlp_conf["num_layers"],
        dropout_rate=mlp_conf["dropout_rate"],
        normalization=mlp_conf.get("normalization", "layer"),
        activation=mlp_conf.get("activation", "relu"),
    )
    summary_net.build((None, n_cls))
    summary_net.summary()

    lr = float(mlp_conf["learning_rate"])
    if mlp_conf.get("lr_schedule", "cosine") == "constant":
        lr_schedule = lr
    else:
        warmup_steps = mlp_conf.get("lr_warmup_steps", 0)
        lr_alpha = mlp_conf.get("lr_alpha", 0.0)
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=lr,
            decay_steps=n_steps,
            alpha=lr_alpha,
            warmup_steps=warmup_steps,
        )
    weight_decay = mlp_conf.get("weight_decay", None)
    if weight_decay is not None:
        optimizer = tf.keras.optimizers.AdamW(lr_schedule, weight_decay=float(weight_decay))
    else:
        optimizer = tf.keras.optimizers.Adam(lr_schedule)

    summary_dir = os.path.join(pred_dir, "network/history")
    model = GridLossModel(
        summary_net,
        n_side=None,
        indices=None,
        optimizer=optimizer,
        checkpoint_dir=os.path.join(pred_dir, "network/checkpoint"),
        summary_dir=summary_dir,
        restore_checkpoint=args.restore_checkpoint,
        xla=mlp_conf.get("xla", False),
    )

    model.setup_grid_loss_step(
        batch_size=batch_size,
        dim_theta=n_params,
        loss="mutual_info",
        dim_x=n_cls,
        dim_summary=n_summary,
        mutual_info_estimator=vmim_conf["estimator"],
        clip_by_global_norm=mlp_conf.get("clip_by_global_norm", 1.0),
        mutual_info_kwargs={
            "density_estimator": vmim_conf["density_estimator"],
            "num_hidden_layers": vmim_conf.get("num_hidden_layers", 2),
            "num_hidden_units": vmim_conf.get("num_hidden_units", 128),
            "activation": vmim_conf.get("activation", "relu"),
            "full_covariance": vmim_conf.get("full_covariance", True),
            "num_components": vmim_conf.get("num_components", 4),
            "num_layers": vmim_conf.get("num_layers", 4),
            "scale_eps": float(vmim_conf.get("scale_eps", 1e-5)),
            "log_scale_clip": float(vmim_conf.get("log_scale_clip", 5.0)),
        },
    )

    tb_writer = tf.summary.create_file_writer(summary_dir)

    es_conf = mlp_conf.get("early_stopping", {})
    early_stopper = (
        EarlyStopper(
            patience=es_conf.get("patience", 10),
            min_delta=float(es_conf.get("min_delta", 1e-4)),
            min_steps=es_conf.get("min_steps", 0),
        )
        if es_conf
        else None
    )

    # --- test-set tensors (noiseless log-Cls) for DC evaluation during training ---
    cls_test_dc = np.array(out_dict["grid/cls/test"], dtype=np.float32)
    grid_cosmos_dc = np.array(out_dict["grid/cosmos/test"], dtype=np.float32)
    _n_dc = min(2048, len(cls_test_dc))
    cls_test_dc_tf = tf.constant(cls_test_dc[:_n_dc])
    grid_cosmos_dc_tf = tf.constant(grid_cosmos_dc[:_n_dc])

    train_steps, train_losses = [], []
    vali_steps, vali_losses_history, vali_dc_history = [], [], []

    for i, (cl_batch, cosmo_batch) in tqdm(enumerate(cl_dset_train), total=n_steps + 1):
        if i > n_steps:
            break
        loss = model.grid_train_step(cl_batch, cosmo_batch)

        if i % log_every == 0:
            train_loss_val = float(loss.numpy())
            train_steps.append(i)
            train_losses.append(train_loss_val)
            with tb_writer.as_default():
                tf.summary.scalar("loss/train", train_loss_val, step=i)
            tb_writer.flush()

        if i > 0 and i % vali_every == 0:
            vali_loss_vals = [
                float(model.vali_loss_fn(model(cl_v, training=False), cosmo_v).numpy())
                for cl_v, cosmo_v in cl_dset_test
            ]
            vali_loss = np.mean(vali_loss_vals)
            vali_steps.append(i)
            vali_losses_history.append(vali_loss)
            dc_preds = tf.concat(
                [model(cls_test_dc_tf[j : j + batch_size], training=False)
                 for j in range(0, _n_dc, batch_size)],
                axis=0,
            )
            dc_val = float(distance_correlation(dc_preds, grid_cosmos_dc_tf, training=False).numpy())
            vali_dc_history.append(dc_val)
            with tb_writer.as_default():
                tf.summary.scalar("loss/vali", vali_loss, step=i)
                tf.summary.scalar("loss/vali_dc", dc_val, step=i)
            tb_writer.flush()
            tqdm.write(f"step {i:>7d}  vali_loss = {vali_loss:.4f}  vali_dc = {dc_val:.4f}")

            if early_stopper is not None:
                improved = early_stopper.update(vali_loss)
                if improved:
                    model.save_model()
                    tqdm.write(f"  -> new best vali_loss={vali_loss:.4f}, saved checkpoint")
                if early_stopper.should_stop(i):
                    tqdm.write(
                        f"Early stopping at step {i} "
                        f"(best={early_stopper.best_loss:.4f}, wait={early_stopper.wait})"
                    )
                    break

    if early_stopper is None:
        model.save_model()
    else:
        model.restore_model()

    _save_loss_curve(
        pred_dir=pred_dir,
        pred_file=pred_file,
        train_steps=train_steps,
        train_losses=train_losses,
        vali_steps=vali_steps,
        vali_losses=vali_losses_history,
        vali_dc=vali_dc_history,
        log_every=log_every,
    )

    # --- evaluate on test set (directly from out_dict, matching the notebook) ---
    print("Evaluating on test set...")
    cls_test = cls_test_dc  # already extracted above (noiseless log-Cls, full test set)
    grid_cosmos = grid_cosmos_dc

    grid_preds = np.concatenate(
        [
            model(tf.constant(cls_test[i : i + batch_size]), training=False).numpy()
            for i in range(0, len(cls_test), batch_size)
        ],
        axis=0,
    )

    with h5py.File(pred_file, "w") as f:
        f.create_dataset("grid/preds/test", data=grid_preds)
        f.create_dataset("grid/cosmos/test", data=grid_cosmos)

    print(f"Saved {len(grid_preds)} test predictions to {pred_file}")

    # --- named grid observations (one per unique cosmology from test set) ---
    if args.include_grid:
        n_perms = msfm_conf["analysis"]["grid"].get("n_perms_per_cosmo", 1)
        n_patches = msfm_conf["analysis"].get("n_patches", 1)
        stride = n_perms * n_patches
        n_grid_obs = args.n_grid_examples

        for k in range(n_grid_obs):
            i = k * stride
            if i >= len(grid_preds):
                print(f"Only {len(grid_preds)} test examples; stopping grid obs at k={k}")
                break
            obs_label = f"grid_{i}"
            evaluation.append_obs_to_file(pred_file, f"obs/preds/{obs_label}", grid_preds[i])
            evaluation.append_obs_to_file(pred_file, f"obs/cosmos/{obs_label}", grid_cosmos[i])

        print(f"Saved {min(n_grid_obs, len(grid_preds) // stride)} grid observations")

    # --- benchmark fiducial observation ---
    if args.include_bench:
        try:
            _evaluate_bench_fidu_cls(
                model=model,
                pred_file=pred_file,
                data_dir=args.data_dir,
                msfm_conf=msfm_conf,
                dlss_conf=dlss_conf,
                params=params,
                batch_size=batch_size,
                with_lensing=with_lensing,
                with_clustering=with_clustering,
                with_cross_z=with_cross_z,
                with_cross_probe=with_cross_probe,
                ggl_only=ggl_only,
            )
        except Exception as e:
            print(f"WARNING: bench_fidu evaluation failed ({e}), skipping")

    # --- DES Y3 real-data observation ---
    if args.include_des:
        try:
            _evaluate_des_y3(
                model=model,
                msfm_conf=msfm_conf,
                dlss_conf=dlss_conf,
                data_dir=args.data_dir,
                pred_file=pred_file,
                with_lensing=with_lensing,
                with_clustering=with_clustering,
                with_cross_z=with_cross_z,
                with_cross_probe=with_cross_probe,
                ggl_only=ggl_only,
            )
        except Exception as e:
            print(f"WARNING: DES Y3 evaluation failed ({e}), skipping")


def _evaluate_bench_fidu_cls(
    model,
    pred_file,
    data_dir,
    msfm_conf,
    dlss_conf,
    params,
    batch_size,
    with_lensing,
    with_clustering,
    with_cross_z,
    with_cross_probe,
    ggl_only,
):
    from msfm.utils import parameters as msfm_params

    print("Evaluating bench_fidu...")
    obs_file = os.path.join(data_dir, "obs", "fiducial_bench_obs_maps.h5")
    with h5py.File(obs_file, "r") as f_in:
        obs_cls_raw = f_in["obs/cls_raw"][:]
    print(f"obs_cls_raw.shape = {obs_cls_raw.shape}")

    obs_cl = preprocessing.get_preprocessed_cl_observation(
        obs_cl=obs_cls_raw,
        msfm_conf=msfm_conf,
        dlss_conf=dlss_conf,
        base_dir=data_dir,
        nest_in=False,
        with_lensing=with_lensing,
        with_clustering=with_clustering,
        with_cross_z=with_cross_z,
        with_cross_probe=with_cross_probe,
        ggl_only=ggl_only,
        apply_log=True,
        standardize=False,
        make_plot=False,
    )
    obs_cl = np.squeeze(obs_cl)

    fidu_preds = np.concatenate(
        [
            model(tf.constant(obs_cl[i : i + batch_size], dtype=tf.float32), training=False).numpy()
            for i in range(0, len(obs_cl), batch_size)
        ],
        axis=0,
    )

    fiducial_cosmo = msfm_params.get_fiducials(params, msfm_conf)

    evaluation.append_obs_to_file(pred_file, "obs/preds/bench_fidu_stack", fidu_preds)
    evaluation.append_obs_to_file(pred_file, "obs/preds/bench_fidu_mean", np.mean(fidu_preds, axis=0))
    evaluation.append_obs_to_file(pred_file, "obs/cosmos/bench_fidu", fiducial_cosmo)
    print(f"Saved bench_fidu ({len(fidu_preds)} realizations) to {pred_file}")


def _save_loss_curve(pred_dir, pred_file, train_steps, train_losses, vali_steps, vali_losses, log_every, vali_dc=None):
    train_steps = np.array(train_steps)
    train_losses = np.array(train_losses)
    vali_steps = np.array(vali_steps)
    vali_losses = np.array(vali_losses)
    vali_dc = np.array(vali_dc) if vali_dc else np.array([])

    with h5py.File(pred_file, "a") as f:
        for key, arr in [
            ("loss/train_steps", train_steps),
            ("loss/train_losses", train_losses),
            ("loss/vali_steps", vali_steps),
            ("loss/vali_losses", vali_losses),
            ("loss/vali_dc", vali_dc),
        ]:
            if key in f:
                del f[key]
            f.create_dataset(key, data=arr)

    has_vali = len(vali_steps) > 0
    has_dc = len(vali_dc) > 0
    n_panels = 1 + int(has_dc)
    fig, axes = plt.subplots(1, n_panels, figsize=(8 * n_panels, 4))
    if n_panels == 1:
        axes = [axes]

    skip = max(1, 1000 // log_every)
    ax = axes[0]
    ax.plot(train_steps[skip:], train_losses[skip:], lw=0.8, label="train")
    if has_vali:
        ax.plot(vali_steps, vali_losses, lw=1.5, marker="o", ms=3, label="vali")
    ax.set_xlabel("step")
    ax.set_ylabel("MI loss")
    ax.legend()

    if has_dc:
        ax2 = axes[1]
        ax2.plot(vali_steps, vali_dc, lw=1.5, marker="o", ms=3, color="C2", label="vali DC")
        ax2.set_xlabel("step")
        ax2.set_ylabel("distance correlation (lower = more correlated)")
        ax2.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(pred_dir, "loss_curve.png"), dpi=150)
    plt.close(fig)
    print(f"Saved loss curve to {pred_dir}/loss_curve.png")


def _evaluate_des_y3(
    model,
    msfm_conf,
    dlss_conf,
    data_dir,
    pred_file,
    with_lensing,
    with_clustering,
    with_cross_z,
    with_cross_probe,
    ggl_only=False,
):
    from msfm.utils import catalog

    print("Building DES Y3 maps from catalogs...")
    wl_gamma_map, _ = catalog.build_metacal_map_from_cat(msfm_conf)
    gc_count_map = catalog.build_maglim_map_from_cat(msfm_conf)

    print("Computing DES Y3 binned Cls...")
    des_cl = preprocessing.get_preprocessed_cl_observation(
        wl_gamma_map=wl_gamma_map,
        gc_count_map=gc_count_map,
        msfm_conf=msfm_conf,
        dlss_conf=dlss_conf,
        base_dir=data_dir,
        nest_in=False,
        with_lensing=with_lensing,
        with_clustering=with_clustering,
        with_cross_z=with_cross_z,
        with_cross_probe=with_cross_probe,
        ggl_only=ggl_only,
        apply_log=True,
        standardize=False,
        make_plot=False,
        apply_maglim_sys_map=True,
    )

    des_pred = model(tf.constant(des_cl, dtype=tf.float32), training=False).numpy()
    evaluation.append_obs_to_file(pred_file, "obs/preds/DESy3", des_pred)

    # des_cl = preprocessing.get_preprocessed_cl_observation(
    #     wl_gamma_map=wl_gamma_map,
    #     gc_count_map=gc_count_map,
    #     msfm_conf=msfm_conf,
    #     dlss_conf=dlss_conf,
    #     base_dir=data_dir,
    #     nest_in=False,
    #     with_lensing=with_lensing,
    #     with_clustering=with_clustering,
    #     with_cross_z=with_cross_z,
    #     with_cross_probe=with_cross_probe,
    #     apply_log=True,
    #     standardize=False,
    #     make_plot=False,
    #     apply_maglim_sys_map=False,
    # )

    # des_pred = model(tf.constant(des_cl, dtype=tf.float32), training=False).numpy()
    # evaluation.append_obs_to_file(pred_file, "obs/preds/DESy3_no_sys", des_pred)
    # print("Saved DES Y3 prediction")


if __name__ == "__main__":
    main()
