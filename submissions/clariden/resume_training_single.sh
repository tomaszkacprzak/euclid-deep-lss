#!/bin/bash
#SBATCH --account=a0158
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-task=4
#SBATCH --job-name=training
#SBATCH --output=/users/athomsen/dlss/repos/y3-deep-lss/submissions/clariden/slurm/slurm-%j.out

REPOS="/users/athomsen/dlss/repos"
STRATEGY="mirrored"
LOSS="mutual_info"

VERSION="v16"
SUBVERSION="default"
# SUBVERSION="no_sc"
# SUBVERSION="rot_in_place"

# MODEL="40Mpc"
MODEL="v5"

# PROBE="lensing"
PROBE="clustering"
# PROBE="cross"
# PROBE="combined"

MYSCRATCH="/iopsstor/scratch/cscs/athomsen"
INPUT="$MYSCRATCH/deep_lss/data/$VERSION/$SUBVERSION"
OUTPUT="$MYSCRATCH/deep_lss/runs/$VERSION/$SUBVERSION/maps/$PROBE"
LOG="$OUTPUT/$MODEL/logs/"$STRATEGY"_"$SLURM_JOB_ID""

TRAIN_TFR="$INPUT/tfrecords/grid/DESy3_grid_dmb_????.tfrecord"
GRID_EVAL_TFR=$TRAIN_TFR

export WANDB_API_KEY=$(awk '/password/ {print $2}' ~/.netrc)

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TF_NUM_INTRAOP_THREADS=${SLURM_CPUS_PER_TASK}

echo $MODEL

srun --environment=tensorflow --gpu-bind=none --output=""$LOG"_training.log" \
    python $REPOS/y3-deep-lss/deep_lss/apps/run_training.py \
        --dir_base=$OUTPUT \
        --dir_model=$MODEL \
        --loss_function=$LOSS \
        --train_tfr_pattern=$TRAIN_TFR \
        --grid_vali_tfr_pattern=$GRID_EVAL_TFR \
        --dist_strategy="$STRATEGY" \
        --restore_checkpoint \
        --wandb

srun --environment=tensorflow --gpu-bind=none --output=""$LOG"_inference.log" \
    python $REPOS/y3-deep-lss/deep_lss/apps/run_evaluation.py \
        --dist_strategy="$STRATEGY" \
        --grid_vali_tfr_pattern=$GRID_EVAL_TFR
