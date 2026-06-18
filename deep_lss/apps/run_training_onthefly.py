# Copyright (C) 2022 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created February 2023
Author: Arne Thomsen

Train the DeepSphere graph neural networks at the fiducial cosmology and its perturbations using the information
maximizing loss to find an informative summary statistic.

Meant for the GPU nodes of the Perlmutter cluster at NERSC.
"""

import os, sys, threading, warnings
import torch

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("once", category=UserWarning)

import argparse, yaml, wandb, shutil

from datetime import datetime
from time import time
from contextlib import nullcontext

from msfm.onthefly_physics.onthefly_linear import OntheflyPhysicsModelLinear
from msfm.utils import logger, input_output, files, parameters

from deep_lss.utils import distribute, configuration, evaluation, optimization
from deep_lss.models.grid_model import GridLossModel
from deep_lss.nets import NETWORKS
from deep_lss.nets.regression_head import get_cls_embedding_layers

LOGGER = logger.get_logger(__file__)

# Keys present in dlss.yaml dset.common that are only meaningful for Cls (2pt) training
# and unknown to FiducialPipeline / GridPipeline — strip them before splatting into pipe_kwargs.
_CLS_ONLY_KEYS = frozenset({"with_cross_z", "with_cross_probe", "ggl_only"})


def setup():
    description = "Train the specified network at the fiducial cosmology."
    parser = argparse.ArgumentParser(description=description, add_help=True)

    parser.add_argument("-v", "--verbosity",
        type=str,
        default="info",
        choices=("critical", "error", "warning", "info", "debug"),
        help="logging level",
    )
    parser.add_argument("--loss_function",
        type=str,
        default=None,
        choices=["mse", "likelihood", "mutual_info"],
        help="loss function to train with. If omitted, read from loss_function key in the loss config.",
    )
    parser.add_argument("--dist_strategy",
        choices=[None, "ddp"],
        default=None,
        help="distribution strategy, use None to run locally",
    )
    parser.add_argument("--training_record_pattern",
        type=str,
        required=True,
        help="input root dir of the fiducial or grid data vectors (training)",
    )
    parser.add_argument("--validation_record_pattern",
        type=str,
        default=None,
        help="input root dir of the grid data vectors (validation)",
    )
    parser.add_argument("--evaluation_record_pattern",
        type=str,
        default=None,
        help="input root dir of the grid data vectors (evaluation)",
    )
    parser.add_argument("--dir_base",
        type=str,
        default=None,
        help="base dir where the models are saved. If None, a dir within the repo is generated according to the config",
    )
    parser.add_argument("--dir_model",
        type=str,
        default=None,
        help="dir where the model summaries and checkpoints are saved. If None, a dir is generated according to the"
        " current date and time. This dir is appended to the dir_base as a relative path. Passing an absolute path"
        " overrides this.",
    )
    parser.add_argument("--net_config",
        type=str,
        default="config/resnet_vanilla.yaml",
        help=(
            "configuration .yaml file of the model to be trained. None can only be provided if there's a config in"
            " the dir_model and restore_checkpoint is true."
        ),
    )
    parser.add_argument("--probes_config", 
        type=str, 
        default=None, 
        help="probe/parameter config (configs/probes/)"
    )
    parser.add_argument("--scales_config", 
        type=str, 
        default=None, 
        help="scale-cut config (configs/scales/)"
    )
    parser.add_argument("--loss_config", 
        type=str, 
        default=None, 
        help="loss function config (configs/loss/)"
    )
    parser.add_argument("--data_config", 
        type=str, 
        default=None, 
        help="train/test split config (configs/data/)"
    )
    parser.add_argument("--msfm_config",
        type=str,
        default=None,
        help=(
            "configuration .yaml file of the multiprobe-simulation-forward-model pipeline. None means that the"
            " standard configuration file in configs/config.yaml relative to the msfm repo is loaded."
        ),
    )
    parser.add_argument("--restore_checkpoint",
        action="store_true",
        help=(
            "restore the model from a checkpoint instead of initializing it from scratch."
            " Additionally, the configs are loaded from the path in this case"
        ),
    )
    parser.add_argument("--evaluate_training_set", 
        action="store_true", 
        help="evaluate the training set"
    )
    parser.add_argument("--slurm_output", 
        type=str, 
        default=None, 
        help="path to the slurm output file"
    )
    parser.add_argument("--debug", 
        action="store_true", 
        help="activate debug mode"
    )
    parser.add_argument("--profile", 
        action="store_true", 
        help="run the profiler"
    )
    parser.add_argument("--mixed_precision", 
        action="store_true", 
        help="use mixed precision training"
    )
    parser.add_argument("--mixed_precision_dtype",
        type=str,
        default="float16",
        choices=("float16", "bfloat16"),
        help="mixed precision dtype to use when --mixed_precision is enabled",
    )
    parser.add_argument("--summary_every",
        type=int,
        default=1,
        help="log step_time and global_step summaries every N training steps (set to 1 to keep previous behavior)",
    )
    parser.add_argument("--wandb", 
        action="store_true", 
        help="log to weights & biases, otherwise log to tensorboard"
    )
    parser.add_argument("--wandb_tags", 
        nargs="+", 
        type=str, 
        default=None, 
        help="tags for weights & biases"
    )
    parser.add_argument("--wandb_notes", 
        type=str, 
        default=None, 
        help="notes for weights & biases (longer than tags)"
    )
    parser.add_argument("--wandb_sweep_id", 
        type=str, 
        default=None, 
        help="id of the sweep. If None, no sweep is used"
    )
    args, _ = parser.parse_known_args()

    if args.summary_every < 1:
        raise ValueError(f"summary_every must be >= 1, got {args.summary_every}")

    # set up directories
    file_dir = os.path.dirname(__file__)
    args.repo_dir = os.path.abspath(os.path.join(file_dir, "../.."))

    if args.dir_base is None:
        args.dir_base = os.path.join(args.repo_dir, "run_files")
        os.makedirs(args.dir_base, exist_ok=True)
        LOGGER.info(f"Created base directory {args.dir_base}")

    if args.slurm_output is not None:
        args.slurm_output = os.path.abspath(args.slurm_output)

    # print arguments
    logger.set_all_loggers_level(args.verbosity)
    for key, value in vars(args).items():
        LOGGER.info(f"{key} = {value}")

    # set mixed precision 
    if args.mixed_precision:
        policy_name = f"mixed_{args.mixed_precision_dtype}"
        LOGGER.warning(f"Using mixed precision policy {policy_name}")
        torch.set_float32_matmul_precision(policy_name)

    # handle required configs
    if not args.restore_checkpoint:
        for flag in ("probes_config", "loss_config", "data_config"):
            if getattr(args, flag) is None:
                parser.error(f"--{flag} is required for a fresh run")

    return args


def training():

    LOGGER.timer.start("main")
    args = setup()

    ###############################################################
    ###############################################################
    ##
    ## config and restore
    ##
    ###############################################################
    ###############################################################
    
    # initialize a fresh model
    if not args.restore_checkpoint:

        # load the configs
        net_conf = input_output.read_yaml(os.path.join(args.repo_dir, args.net_config))
        dlss_conf = configuration.read_split_configs(args.probes_config, args.scales_config)
        loss_conf = input_output.read_yaml(args.loss_config)
        data_conf = input_output.read_yaml(args.data_config)
        msfm_conf = files.load_config(args.msfm_config)
        loss_function = loss_conf.get("loss_function")
        LOGGER.info(f"Loaded configs from the provided paths")

        if args.dir_model is None:
            net_name = net_conf["name"]
            now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            args.dir_model = f"{now}_{net_name}"
            LOGGER.info(f"Created model directory {args.dir_model}")

        # make output directory
        dir_model = os.path.join(args.dir_base, args.dir_model)
        os.makedirs(dir_model, exist_ok=True)
        LOGGER.info(f"Created output directory {dir_model}")

        # runtime metadata (kept as its own top-level block in the saved config)
        run_conf = {
            "dir_model": dir_model,
            "dir_log": args.slurm_output,
            "loss_func": loss_function,
            "dist_strategy": args.dist_strategy,
        }

        # save the configs as a single nested mapping
        with open(os.path.join(dir_model, "configs.yaml"), "w") as f:
            yaml.dump(
                {
                    "net": net_conf,
                    "dlss": dlss_conf,
                    "loss": loss_conf,
                    "data": data_conf,
                    "msfm": msfm_conf,
                    "run": run_conf,
                },
                f,
            )

    # restore a saved model
    elif args.restore_checkpoint and (args.dir_model is not None):

        # make output directory
        dir_model = os.path.join(args.dir_base, args.dir_model)
        os.makedirs(dir_model, exist_ok=True)
        LOGGER.info(f"Created output directory {dir_model}")

        # load the configs (migrates a legacy multi-document stream if needed)
        conf = configuration.load_run_configs(os.path.join(dir_model, "configs.yaml"))
        net_conf = conf["net"]
        dlss_conf = conf["dlss"]
        loss_conf = conf["loss"]
        data_conf = conf["data"]
        msfm_conf = conf["msfm"]
        loss_function = conf["run"]["loss_func"]
        LOGGER.info(f"Loaded configs from the model directory")

    else:
        raise ValueError(f"Can't restore the model from an unspecified dir_model")

    # to be read by the evaluation script
    job_id = os.environ["SLURM_JOB_ID"]
    if job_id is not None:
        temp_file = f"./.env_var/id_{job_id}.txt"
        os.makedirs(os.path.dirname(temp_file), exist_ok=True)
        LOGGER.info(f"Writing the model directory to {temp_file}")
        with open(temp_file, "w") as f:
            f.write(dir_model)

    ###############################################################
    ###############################################################
    ##
    ## weights and biases
    ##
    ###############################################################
    ###############################################################

    if args.wandb:
        group_name = wandb.util.generate_id()

        # check if there's an existing run ID to resume
        wandb_id_file = os.path.join(dir_model, "wandb_run_id.txt")
        existing_run_id = None

        if os.path.exists(wandb_id_file) and args.restore_checkpoint:
            with open(wandb_id_file, "r") as f:
                existing_run_id = f.read().strip()
            LOGGER.info(f"Found existing wandb run ID: {existing_run_id}")

        if existing_run_id:
            wandb_run = wandb.init(
                id=existing_run_id,
                resume="allow",
                project="euclid-deep-lss",
                dir=dir_model,
                group=group_name,
                job_type="training",
                # make sure that wandb logs to the cloud
                mode="online",
                force=True,
                # to be able to log within graph mode
                sync_tensorboard=True,
                # additional metadata
                tags=args.wandb_tags,
                notes=args.wandb_notes,
            )
            LOGGER.info(f"Resumed wandb run: {existing_run_id}")
        else:
            wandb_run = wandb.init(
                project="euclid-deep-lss",
                dir=dir_model,
                group=group_name,
                job_type="training",
                mode="online",
                force=True,
                sync_tensorboard=True,
                tags=args.wandb_tags,
                notes=args.wandb_notes,
            )
            LOGGER.info(f"Created new wandb run: {wandb_run.id}")

            # Save the run ID for future resumption
            with open(wandb_id_file, "w") as f:
                f.write(wandb_run.id)

        if args.wandb_sweep_id is not None:

            # in the wandb sweep config, the hyperparameters are defined like net.optimization.optimizer, while the
            # .yaml config files are structured as nested dictionaries
            nested_hyperparam_conf = configuration.convert_dotted_to_nested_dict(wandb_run.config)

            # dict.update() would discard branches that are not present in the update dict
            net_conf = configuration.update_nested_dict(net_conf, nested_hyperparam_conf["net"])

        # only update the config here instead of in the init so that possible changes by a sweep agent are included
        wandb_run.config.setdefaults({"msfm": msfm_conf, "dlss": dlss_conf, "net": net_conf})

        wandb.define_metric("train_step")
        for prefix in ("loss/*", "schedule/*", "learning_rate", "global_grad_norm*", "step_time", "data_time", "compute_time", "z_bank/*", "z_invariance/*"):
            wandb.define_metric(prefix, step_metric="train_step")

        LOGGER.info(f"Initialized weights & biases to {dir_model}")
        

    
    ###############################################################
    ###############################################################
    ##
    ## constants
    ##
    ###############################################################
    ###############################################################

    # set up subdirectories
    checkpoint_dir = os.path.abspath(os.path.join(dir_model, "checkpoint"))
    os.makedirs(checkpoint_dir, exist_ok=True)
    summary_dir = os.path.abspath(os.path.join(dir_model, "summary"))
    os.makedirs(summary_dir, exist_ok=True)

    # constants: msfm
    n_side = msfm_conf["analysis"]["n_side"]
    data_vec_pix, _, _, _ = files.load_pixel_file(msfm_conf)

    smooth_nside = net_conf["network"].get("smooth_nside", None)
    if smooth_nside is not None and smooth_nside < n_side:
        smooth_indices, parent_output_idx = configuration.get_smooth_nside_indices(data_vec_pix, n_side, smooth_nside)
        LOGGER.info(f"Using smooth_nside={smooth_nside}: {len(data_vec_pix)} → {len(smooth_indices)} pixels")
    else:
        smooth_nside = n_side
        smooth_indices = data_vec_pix
        parent_output_idx = None

    # constants: deep_lss
    params = dlss_conf["dset"]["training"]["params"]
    n_params = len(params)
    LOGGER.info(f"Training with respect to the {n_params} parameters {params}")

    with_lensing = dlss_conf["dset"]["common"]["with_lensing"]
    with_clustering = dlss_conf["dset"]["common"]["with_clustering"]
    with_cross = dlss_conf["dset"]["common"].get("with_cross", False)
    return_cls = "cls_n_bins" in net_conf["network"]
    if return_cls:
        LOGGER.warning("cls_n_bins detected in net_conf['network'] — will build MapsPlusCLSNetwork")

    # constants: network
    n_steps = net_conf["training"]["n_steps"]
    output_every = net_conf["training"]["output_every"]
    checkpoint_every = net_conf["training"]["checkpoint_every"]
    vali_every = net_conf["training"]["vali_every"]
    eval_every = net_conf["training"]["eval_every"]

    # constants: miscellaneous
    training_type = "grid"
    smoothing_kwargs = configuration.get_smoothing_kwargs(
        loss_function, msfm_conf, dlss_conf, net_conf, dir_base=dir_model
    )

    # loss function selection
    dset_kwargs = {**net_conf["dset"]["training"]["common"], **data_conf}
    noise_kwargs = {}

    if loss_function == "likelihood":
        n_output = n_params + n_params * (n_params + 1) // 2
    elif loss_function == "mse":
        n_output = n_params
    elif loss_function == "mutual_info":
        n_output = loss_conf["mutual_info_loss"]["dim_summary_fac"] * n_params

    # constants: redshift bins
    try:
        n_z_bins = len(dset_kwargs["z_bin_inds"])
    except (KeyError, TypeError):
        n_z_bins = 0
        if with_lensing:
            n_z_bins += len(msfm_conf["survey"]["WL"]["z_bins"])
        if with_clustering:
            n_z_bins += len(msfm_conf["survey"]["GC"]["z_bins"])
        if with_cross:
            n_z_bins += len(msfm_conf["survey"]["WL"]["z_bins"]) * len(msfm_conf["survey"]["GC"]["z_bins"])



    ###############################################################
    ###############################################################
    ##
    ## dataset
    ##
    ###############################################################
    ###############################################################

    # loader
    Pipeline = OntheflyPhysicsModelLinear
    # TODO: implement GridLossModel in pytorch
    Model = GridLossModel
    dset_kwargs.update(net_conf["dset"]["training"]["grid"])
    local_batch_size = dset_kwargs["local_batch_size"]
    effective_local_batch_size = local_batch_size


    # dataset
    LOGGER.warning(f"Training set")
    pipe_kwargs = {k: v for k, v in {**dlss_conf["dset"]["common"], **dlss_conf["dset"]["training"], **noise_kwargs}.items()
                   if k not in _CLS_ONLY_KEYS}
    pipe_kwargs["return_maps"] = True
    pipe_kwargs["return_cls"] = return_cls
    train_pipeline = Pipeline(conf=msfm_conf, **pipe_kwargs)

    # TODO: implement get_dset for onthefly pipeline
    dset = train_pipeline.get_dset(
            record_pattern=args.training_record_pattern,
            **dset_kwargs,
            # nside downsampling
            downsample_nside=smooth_nside if parent_output_idx is not None else None,
            parent_output_idx=parent_output_idx,
    )

    dist_iter = iter(dset)


    ###############################################################
    ###############################################################
    ##
    ## network
    ##
    ###############################################################
    ###############################################################

    # network, create all of the variables within the strategy's scope, such that they are mirrored
    # TODO: implement networks in pytorch
    net_class = NETWORKS[net_conf["network"]["name"]]
    net_spec = net_class(out_features=n_output, 
                         smoothing_kwargs=smoothing_kwargs, 
                         **net_conf["network"]["kwargs"])
    LOGGER.info(f"Loaded a network specification of type {net_class}")
    LOGGER.info(f"Network kwargs including regularization: {net_conf['network']['kwargs']}")

    optimizer = optimization.get_optimizer(net_conf, loss_function, args.restore_checkpoint)

    network = net_spec.get_layers()
    model = Model(
        network=network,
        n_side=smooth_nside,
        indices=smooth_indices,
        n_neighbors=net_conf["network"]["n_neighbors"],
        z_bank_size=net_conf["network"]["z_bank_size"],
        max_checkpoints=net_conf["network"]["max_checkpoints"],
        optimizer=optimizer,
        input_shape=(None, len(smooth_indices), n_z_bins),
        max_batch_size=effective_local_batch_size,
        checkpoint_dir=checkpoint_dir,
        summary_dir=summary_dir,
        restore_checkpoint=args.restore_checkpoint,
        summary_every=args.summary_every,
    )

    ###############################################################
    ###############################################################
    ##
    ## loss function
    ##
    ###############################################################
    ###############################################################
    
    if loss_function == "mse":

        loss_kwargs = {}

    elif loss_function == "likelihood":
        if not args.restore_checkpoint:

            lambda_tikhonov_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=loss_conf["likelihood_loss"]["lambda_tikhonov_decay_steps"],
                eta_min=loss_conf["likelihood_loss"]["lambda_tikhonov_init"] * 0.0,
            )
            lambda_tikhonov = torch.tensor(lambda_tikhonov_schedule(0), dtype=torch.float32)

        else:
            lambda_tikhonov = torch.tensor(0.0, dtype=torch.float32)
                
        loss_kwargs = {
            "lambda_tikhonov": lambda_tikhonov,
            "img_summary": loss_conf["likelihood_loss"]["img_summary"],
        }
    
    elif loss_function == "mutual_info":

        loss_kwargs = {
                "dim_summary": n_output,
                **loss_conf["mutual_info_loss"]["regu"],
                "mutual_info_estimator": loss_conf["mutual_info_loss"]["estimator"],
                "mutual_info_kwargs": loss_conf["mutual_info_loss"]["kwargs"],
        }

    else:
        raise ValueError(f"Invalid loss function: {loss_function}")

    # TODO: implement setup_grid_loss_step for onthefly pipeline with pytorch
    model.setup_grid_loss_step(
        loss=loss_function,
        batch_size=local_batch_size,
        dim_theta=n_params,
        dim_x=None if return_cls else len(data_vec_pix),
        dim_channels=None if return_cls else n_z_bins,
        **loss_kwargs,
        **net_conf["optimization"]["gradient_clipping"],
    )



    # validation loss
    if vali_every is not None:

        vali_pipe_kwargs = {k: v for k, v in dlss_conf["dset"]["common"].items() if k not in _CLS_ONLY_KEYS}
        vali_pipe_kwargs["return_maps"] = True
        vali_pipe_kwargs["return_cls"] = return_cls
        vali_dset_kwargs = {**net_conf["dset"]["validation"]["common"], **data_conf}
        vali_dset_kwargs["drop_remainder"] = True
        n_vali_batches = net_conf["dset"]["validation"]["n_batches"]

        def make_validation_loop(dist_dset, step_fn, n_expected, summary_map):
            def validation_loop():
                metrics = [tf.keras.metrics.Mean(), tf.keras.metrics.Mean()]
                for batch_tuple in LOGGER.progressbar(dist_dset, at_level="debug", desc="validation", total=n_expected):
                    vals = step_fn(batch_tuple)
                    for i, v in enumerate(vals):
                        if not tf.math.is_nan(v):
                            metrics[i].update_state(v)
                assert not tf.math.is_nan(
                    metrics[0].result()
                ), "Validation loss is NaN, check the validation batch size as this is likely due to partially empty batches"
                for key, idx in summary_map:
                    model.write_summary(key, metrics[idx].result())
                for m in metrics:
                    m.reset_states()
            return validation_loop


        vali_pipe_kwargs["params"] = dlss_conf["dset"]["eval"]["grid"]["params"]
        vali_dset_kwargs.update(net_conf["dset"]["validation"]["grid"])

        LOGGER.warning(f"Grid validation set")
        n_vali_examples_per_replica = n_vali_batches * vali_dset_kwargs["local_batch_size"] if n_vali_batches is not None else None
        LOGGER.info(
            f"Grid validation: {n_vali_batches} batches × local_batch_size "
            f"{vali_dset_kwargs['local_batch_size']} = "
            f"{n_vali_examples_per_replica} examples/replica, every {vali_every} steps"
        )
        vali_pipeline = Pipeline(conf=msfm_conf, **vali_pipe_kwargs)

        # TODO: finish implementation of validation loop for onthefly pipeline with pytorch


    LOGGER.info(f"Starting training")
    LOGGER.timer.start("training")
    t_prev = time()
    t_accum = 0.0
    t_data_accum = 0.0
    t_compute_accum = 0.0

    for step in LOGGER.progressbar(range(1, n_steps + 1), at_level="info", total=n_steps, desc="training"):

        # train step
        t_data_start = time()
        dv_batch, cl_batch, cosmo_batch, index_batch = next(dist_iter)
        t_data_end = time()
        x_batch = (dv_batch, cl_batch) if return_cls else dv_batch
        loss = model.grid_train_step(x_batch, cosmo_batch)
        t_compute_end = time()

        # likelihood loss
        if args.loss_function == "likelihood" and not args.restore_checkpoint:
            lambda_tikhonov.assign(lambda_tikhonov_schedule(step))
            model.write_summary("schedule/lambda_tikhonov", lambda_tikhonov)

        # output
        if (output_every is not None) and (step % output_every == 0):
            _copy_log(args, dir_model)

        # checkpoint
        if (checkpoint_every is not None) and (step % checkpoint_every == 0):
            model.save_model()

        # validate
        if (vali_every is not None) and (step % vali_every == 0):
            # since at that step, everything should be already traced

            # TODO: finish implementation of validation loop for onthefly pipeline with pytorch
            # validation_loop()
            if model.summary_writer is not None:
                model.summary_writer.flush()

        # evaluate
        if (eval_every is not None) and (step % eval_every == 0):
            train_step = model.get_step()
            LOGGER.info(f"Evaluating the model after a total of {train_step} training steps")

            out_file = None

            # fiducial training
            if args.evaluate_training_set:

                out_file = evaluation.evaluate_grid(
                    model=model,
                    record_pattern=args.training_record_pattern,
                    msfm_conf=msfm_conf,
                    dlss_conf=dlss_conf,
                    net_conf=net_conf,
                    data_conf=data_conf,
                    dir_out=dir_model,
                    file_label=train_step,
                )

            else:
                LOGGER.warning(f"Skipping evaluation of the fiducial training set")

            # grid evaluation
            if args.evaluation_record_pattern is not None:
                out_file = evaluation.evaluate_grid(
                    model=model,
                    record_pattern=args.evaluation_record_pattern,
                    msfm_conf=msfm_conf,
                    dlss_conf=dlss_conf,
                    net_conf=net_conf,
                    data_conf=data_conf,
                    dir_out=dir_model,
                    file_label=train_step,
                )
            else:
                LOGGER.warning(f"Skipping evaluation of the grid evaluation set")

            # log here instead of inside eval to avoid partial duplicate .h5 files
            if args.wandb and (out_file is not None):
                wandb_artifact = wandb.Artifact(
                    name=f"training-predictions-nsteps{train_step}", type="predictions"
                )
                wandb_artifact.add_file(local_path=out_file)
                wandb_run.log_artifact(wandb_artifact)
                LOGGER.info(f"Logged the predictions to weights & biases after step {step}")

        # additional logs
        t_now = time()
        t_accum += t_now - t_prev
        t_data_accum += t_data_end - t_data_start
        t_compute_accum += t_compute_end - t_data_end
        t_prev = t_now
        if step % args.summary_every == 0:
            model.write_summary("step_time", t_accum / args.summary_every)
            model.write_summary("data_time", t_data_accum / args.summary_every)
            model.write_summary("compute_time", t_compute_accum / args.summary_every)
            model.write_summary("global_step", model.get_step())
            t_accum = 0.0
            t_data_accum = 0.0
            t_compute_accum = 0.0

    LOGGER.info(f"Finished training after {n_steps} steps and {LOGGER.timer.elapsed('training')}")

    # finalize EMA weight averaging, if enabled
    inner_optimizer = getattr(optimizer, "inner_optimizer", optimizer)
    ema_finalized = getattr(inner_optimizer, "use_ema", False)
    if ema_finalized:
        LOGGER.info(f"Finalizing EMA weights")
        inner_optimizer.finalize_variable_values(model.trainable_variables)

    # save everything at the end if necessary
    if ema_finalized or ((checkpoint_every is not None) and (step % checkpoint_every != 0)):
        LOGGER.info(f"Creating a final checkpoint")
        model.save_model()
    elif checkpoint_every is not None:
        LOGGER.info(f"A final checkpoint already exists")
    else:
        LOGGER.info(f"No checkpoint has been saved")

    if args.wandb:
        wandb.finish()
    model.delete_temp_summaries()

    LOGGER.info(f"Script completed successfully")
    _copy_log(args, dir_model)


def _copy_log(args, dir_out):
    if args.slurm_output is not None:
        dir_log = os.path.join(dir_out, "logs")
        os.makedirs(dir_log, exist_ok=True)

        file_log = os.path.join(dir_log, os.path.basename(args.slurm_output))
        shutil.copy(args.slurm_output, file_log)


if __name__ == "__main__":
    
    args = setup()

    if args.wandb_sweep_id is None:
        training()
    else:
        wandb.agent(args.wandb_sweep_id, function=training, project="euclid-deep-lss", count=1)
