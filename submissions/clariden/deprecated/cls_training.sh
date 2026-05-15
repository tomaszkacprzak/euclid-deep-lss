#!/bin/bash
#SBATCH --account=a0158
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=450G
#SBATCH --job-name=cls_training
#SBATCH --output=/users/athomsen/dlss/repos/y3-deep-lss/submissions/clariden/slurm/slurm-%j.out

REPOS="/users/athomsen/dlss/repos"
MYSCRATCH="/iopsstor/scratch/cscs/athomsen"

VERSION="v16"
SUBVERSION="rot_in_place"
# PROBE="lensing"
# PROBE="clustering"
# PROBE="cross"
# PROBE="2x2pt"
# PROBE="combined"
PROBE="${PROBE:-lensing}"
# MODEL_VERSION="v3/gmm_fullcov"
# MODEL_VERSION="v3/flow"
MODEL_VERSION="v4"

# MLP_CONFIGS=("default" "lr" "cosine_decay" "small")
MLP_CONFIGS=("default" "large" "dropout" "cosine_decay")
# MLP="default"

INPUT="$MYSCRATCH/deep_lss/data/$VERSION/$SUBVERSION"
OUTPUT="$MYSCRATCH/deep_lss/runs/$VERSION/$SUBVERSION/cls/$PROBE"
DLSS_CONFIG="$REPOS/y3-deep-lss/configs/$VERSION/$SUBVERSION/$PROBE/dlss.yaml"

export SLURM_CPUS_PER_TASK=72
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TF_NUM_INTRAOP_THREADS=${SLURM_CPUS_PER_TASK}

for MLP in "${MLP_CONFIGS[@]}"; do
    MODEL="$MODEL_VERSION/$MLP"
    LOG="$OUTPUT/$MODEL/logs/${SLURM_JOB_ID}"
    mkdir -p "$(dirname "$LOG")"

    srun -N1 --ntasks-per-node=1 --exclusive --gpus-per-task=1 --cpus-per-gpu=72 --mem=110G \
        --environment=tensorflow \
        --output="${LOG}_training.log" \
        python $REPOS/y3-deep-lss/deep_lss/apps/2pt/run_cls_training.py \
            --msfm_config="$REPOS/multiprobe-simulation-forward-model/configs/$VERSION/$SUBVERSION.yaml" \
            --dlss_config="$DLSS_CONFIG" \
            --mlp_config="$REPOS/y3-deep-lss/configs/mlp/${MLP}.yaml" \
            --data_dir="$INPUT" \
            --out_dir="$OUTPUT" \
            --model_name="$MODEL" &
done

wait

echo "Starting Inference..."

for MLP in "${MLP_CONFIGS[@]}"; do
    MODEL="$MODEL_VERSION/$MLP"
    LOG="$OUTPUT/$MODEL/logs/${SLURM_JOB_ID}"

    srun -N1 --ntasks-per-node=1 --exclusive --gpus-per-task=1 --cpus-per-gpu=72 --mem=110G \
        --uenv=pytorch/v2.9.1:v2 --view=default \
        --output="${LOG}_inference.log" \
        bash -c "source ~/dlss/torch_env/bin/activate && python $REPOS/multiprobe-simulation-inference/msi/apps/2pt/run_cls_inference.py \
            --msfm_config=\"$REPOS/multiprobe-simulation-forward-model/configs/$VERSION/$SUBVERSION.yaml\" \
            --dlss_config=\"$DLSS_CONFIG\" \
            --out_dir=\"$OUTPUT\" \
            --model_name=\"$MODEL\"" &
done

wait
