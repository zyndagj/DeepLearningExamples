#!/usr/bin/env bash
# Script to benchmark multi-GPU training performance, with bases precomputation

# CLI args with defaults
#BATCH_SIZE=${1:-240}
BATCH_SIZE=${1:-120}
AMP=${2:-true}
ARGS="--amp $AMP --batch_size $BATCH_SIZE --epochs 6 --use_layer_norm --norm --save_ckpt_path model_qm9.pth --task homo --precompute_bases --seed 42 --benchmark"

for l in 14 28 7; do
unset NCCL_P2P_DISABLE
N=dgx1_n1_g8_l${l}_nvl; echo $N
python -m torch.distributed.run --nnodes=1 --nproc_per_node=gpu --max_restarts 0 --module \
	se3_transformer.runtime.training $ARGS \
	--num_layers ${l} &> ${N}.log
NCCL_P2P_DISABLE=1
N=dgx1_n1_g8_l${l}_pcie; echo $N
python -m torch.distributed.run --nnodes=1 --nproc_per_node=gpu --max_restarts 0 --module \
	se3_transformer.runtime.training $ARGS \
	--num_layers ${l} &> ${N}.log
done
