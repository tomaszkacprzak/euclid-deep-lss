```
# Clariden internal
rsync -ahv /iopsstor/scratch/cscs/athomsen/deep_lss/runs /capstor/store/cscs/swissai/a0158/athomsen/deep_lss

# Perlmutter to MacBook
rsync -ahv --prune-empty-dirs \
  --exclude={"**/wandb/**","**/wandb","**/smoothing/**","**/smoothing"} \
  --include={"*/","*.yaml","*.h5","*.npy","*.pt","*.tf","*.png","*.pdf"} \
  --exclude="*" \
  athomsen@perlmutter-p1.nersc.gov:/pscratch/sd/a/athomsen/deep_lss/runs \
  /Users/arne/data/deep_lss

# Clariden to MacBook
rsync -ahv --prune-empty-dirs \
  --exclude={"**/wandb/**","**/wandb","**/smoothing/**","**/smoothing"} \
  --include={"*/","*.yaml","*.h5","*.npy","*.pt","*.tf","*.png","*.pdf"} \
  --exclude="*" \
  athomsen@clariden:/iopsstor/scratch/cscs/athomsen/deep_lss/runs \
  /Users/arne/data/deep_lss

# Perlmutter to Clariden (from Clariden)
rsync -ahv --prune-empty-dirs \
  --exclude={"**/wandb/**","**/wandb","**/smoothing/**","**/smoothing"} \
  --include={"*/","*.yaml","*.h5","*.npy","*.pt","*.tf","*.png","*.pdf"} \
  --exclude="*" \
  athomsen@perlmutter.nersc.gov:/pscratch/sd/a/athomsen/deep_lss/runs \
  /iopsstor/scratch/cscs/athomsen/deep_lss

rsync -ahv --prune-empty-dirs \
  athomsen@perlmutter.nersc.gov:/pscratch/sd/a/athomsen/deep_lss/runs \
  /iopsstor/scratch/cscs/athomsen/deep_lss
```

# catalogs
```
nohup wget -c \
http://desdr-server.ncsa.illinois.edu/despublic/y3a2_files/y3kp_cats/DESY3_indexcat.h5 \
http://desdr-server.ncsa.illinois.edu/despublic/y3a2_files/y3kp_cats/DESY3_GOLD_2_2.1.h5 \
http://desdr-server.ncsa.illinois.edu/despublic/y3a2_files/y3kp_cats/DESY3_metacal_v03-004.h5 \
http://desdr-server.ncsa.illinois.edu/despublic/y3a2_files/y3kp_cats/DESY3_maglim_redmagic_v0.5.1.h5 \
http://desdr-server.ncsa.illinois.edu/despublic/y3a2_files/y3kp_cats/DESY3_GOLD_2_2.1_DNF.h5 \
http://desdr-server.ncsa.illinois.edu/despublic/y3a2_files/y3kp_cats/DESY3_sompz_v0.50.h5 \
> download.log 2>&1 &
```