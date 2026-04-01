# TensorFlow
Using a (deprecated) NVIDIA container, following https://docs.cscs.ch/software/container-engine/#quick-start.
This is TensorFlow 2.17

## set up GitHub repos
In `${HOME}/dlss`
```
git clone https://github.com/des-science/multiprobe-simulation-forward-model.git
git clone https://github.com/des-science/y3-deep-lss.git
git clone https://github.com/des-science/multiprobe-simulation-inference.git
git clone https://github.com/Arne-Thomsen/deepsphere-cosmo-tf2
```

## set up container
Save as `${HOME}/.edf/tensorflow.toml` referencing the [NGC catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorflow?version=25.02-tf2-py3-igpu):
```
image = "nvcr.io/nvidia/tensorflow:25.02-tf2-py3"
mounts = [
    "${SCRATCH}:${SCRATCH}",
    "${HOME}/dlss:${HOME}/dlss",
    "${HOME}/dlss/container_bashrc:${HOME}/.bashrc"
]
workdir = "${HOME}/dlss"

[env]
BASH_ENV = "${HOME}/.bashrc"

[annotations]
com.hooks.aws_ofi_nccl.enabled = "true"
com.hooks.aws_ofi_nccl.variant = "cuda12"
```

## set up virtual environment
```
# on the login node, following https://docs.astral.sh/uv/#installation
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="${HOME}/dlss/bin" sh

# start interactive session running the container
srun -A a0158 --environment=tensorflow --partition=debug --pty bash

# uv setup
source $HOME/dlss/bin/env
export UV_CACHE_DIR="${SCRATCH}/.cache/uv"

# create and activate virtual environment
uv venv --python $(which python) --system-site-packages --seed --relocatable --link-mode=copy ~/dlss/tf_env
source ~/dlss/tf_env/bin/activate

# for compatibility with tf 2.17
uv pip install tensorflow_probability==0.24 

# install from repos
uv pip install -e ~/dlss/multiprobe-simulation-forward-model
uv pip install -e ~/dlss/y3-deep-lss
uv pip install -e ~/dlss/multiprobe-simulation-inference
uv pip install -e ~/dlss/add_on/deepsphere-cosmo-tf2

# remove pypi version to use the ones from the container
uv pip uninstall tensorflow tensorboard keras numpy scipy numba llvmlite protobuf

# test GPUs
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## set up `.bashrc`
Save as `${HOME}/dlss/tf_env_bashrc`.
```
# add custom persistent bin to the PATH (so 'uv' is always available)
export PATH="${HOME}/dlss/bin:$PATH"

# set the uv cache directory to SCRATCH to protect your HOME quota
export UV_CACHE_DIR="${SCRATCH}/.cache/uv"

# auto-activate the virtual environment
source ${HOME}/dlss/tf_env/bin/activate
```

## set up Jupyter kernel
```
uv pip install ipykernel
python -m ipykernel install ${VIRTUAL_ENV:+--env PATH $PATH --env VIRTUAL_ENV $VIRTUAL_ENV} --user --name="tf_env"
```
