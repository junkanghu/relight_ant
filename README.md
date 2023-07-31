# How to run the training code?

## Not on ant cluster with ddp.
``` shell
DEFAULT_FREE_PORT=9849 &&\
workspace=xxx OMP_NUM_THREADS=96 \
python -m torch.distributed.launch --nproc_per_node 8 \
--master_addr ${MASTER_IP:-127.0.0.1} \
--master_port ${MASTER_PORT:-$DEFAULT_FREE_PORT} \
--nnodes ${NODE_SIZE:-1} \
--node_rank ${NODE_RANK:-0}
```
Here, ```DEFAULT_FREE_PORT``` is set by user. ```workspace``` shoule be the output directory, if not run the code on ant cluster, it should be ```./``` so that the results will be saved under the code directory.

## On ant cluster with ddp.
``` shell
workspace=xxx OMP_NUM_THREADS=96 \
python -m torch.distributed.launch --nproc_per_node 8 \
--master_addr ${MASTER_IP:-127.0.0.1} \
--master_port ${MASTER_PORT:-$DEFAULT_FREE_PORT} \
--nnodes ${NODE_SIZE:-1} \
--node_rank ${NODE_RANK:-0}
```
The differences between running on cluster and lab server: 
1. We don't need to set ```DEFAULT_FREE_PORT```, bacause it is set by cluster by default.
2. We should set the workspace as ```/input/pihuaijin/workspaces/relight```, because the working directory for the code is temporary which will be deleted after the code is finished.

## Without ddp
``` shell
OMP_NUM_THREADS=96 python train.py
```
If the code is ran on the lab server which just saves the output under the code directory, we don't need to explicitly set ```workspace```, the default one is under the code directory. **However, we do need to caution that the output dir should not be ./results but results. See the code in utils workspace_config**.