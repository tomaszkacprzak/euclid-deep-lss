# PyTorch
Using a `uenv` following https://docs.cscs.ch/software/ml/pytorch/#running-pytorch-with-a-uenv.

## set up `uenv`
```
uenv image find

uenv image pull pytorch/v2.9.1:v2
uenv start --view=default pytorch/v2.9.1:v2

# https://docs.cscs.ch/build-install/python/#installing-venv-on-top-of-a-uenv-view
unset PYTHONPATH
export PYTHONUSERBASE="$(dirname "$(dirname "$(which python)")")"
```

## set up virtual environment 
```
# uv setup
export UV_CACHE_DIR="${SCRATCH}/.cache/uv"

# create and activate virtual environment
uv venv --python $(which python) --system-site-packages --seed --relocatable --link-mode=copy ~/dlss/torch_env
source ~/dlss/torch_env/bin/activate

# install from repos
uv pip install -e ~/dlss/repos/multiprobe-simulation-forward-model
uv pip install -e ~/dlss/repos/y3-deep-lss
uv pip install -e ~/dlss/repos/multiprobe-simulation-inference --override <(echo "pandas>=2.1.0")

# remove pypi version to use the ones from the uenv
uv pip uninstall torch sympy networkx mpmath

# test GPUs
python -c "import torch; print(torch.cuda.device_count())"
```

## set up Jupyter kernel
```
uv pip install ipykernel
python -m ipykernel install ${VIRTUAL_ENV:+--env PATH $PATH --env VIRTUAL_ENV $VIRTUAL_ENV} --user --name="torch_env"
```
To launch JupyterLab, specify `pytorch/v2.9.1:v2` in "Custom uenv".
