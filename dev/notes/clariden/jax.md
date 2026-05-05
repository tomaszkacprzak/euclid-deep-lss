# JAX
Using an NVIDIA container, following https://docs.cscs.ch/software/container-engine/#quick-start.

## set up relevant GitHub repos
In `${HOME}/disco/repos`
```
git clone git@github.com:jbucko/DISCOFalcon.git
```

## set up `.bashrc`
Save as `${HOME}/disco/.jax_env_bashrc`.
```
# add custom persistent bin to the PATH (so 'uv' is always available)
export PATH="${HOME}/disco/bin:$PATH"

# set the uv cache directory to SCRATCH to protect your HOME quota
export UV_CACHE_DIR="${SCRATCH}/.cache/uv"

# auto-activate the virtual environment
source ${HOME}/disco/jax_env/bin/activate
```

## set up container
Save as `${HOME}/.edf/jax.toml` referencing the [NGC catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/jax?version=26.03-py3):
```
image = "nvcr.io/nvidia/jax:26.03-py3"
mounts = [
    "${SCRATCH}:${SCRATCH}",
    "${HOME}/disco:${HOME}/disco",
    "${HOME}/disco/.jax_env_bashrc:${HOME}/.bashrc"
]
workdir = "${HOME}/disco"

[env]
PATH = "/users/{$USER}/disco/jax_env/bin:/usr/local/bin:/usr/bin:/bin"
BASH_ENV = "${HOME}/.bashrc"

[annotations]
com.hooks.aws_ofi_nccl.enabled = "true"
com.hooks.aws_ofi_nccl.variant = "cuda12"
```

## set up virtual environment
```
# on a login node, following https://docs.astral.sh/uv/#installation
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="${HOME}/disco/bin" sh

# start interactive session running the container
srun -A a0158 --environment=jax --partition=normal --pty bash

# create and activate virtual environment
uv venv --python $(which python) --system-site-packages --seed --relocatable --link-mode=copy ~/disco/jax_env
source ~/disco/jax_env/bin/activate

# install editable repos 
uv pip install -e ~/disco/repos/TODO

# install jupyter
uv pip install ipykernel
python -m ipykernel install ${VIRTUAL_ENV:+--env PATH $PATH --env VIRTUAL_ENV $VIRTUAL_ENV} --user --name="jax_env"

# test GPUs
python -c "import jax; print(jax.devices())"
```

## JupyterLab
To launch JupyterLab, specify `/users/athomsen/.edf/jax.toml` in "Path to your project's CE toml file".
