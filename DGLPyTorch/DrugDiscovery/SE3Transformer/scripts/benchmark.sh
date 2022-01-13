#!/usr/bin/env bash

###################################
# Helper functions
###################################
PROG=$(basename $0)
function ee {
  echo "[ERROR] $PROG: $@" >&2; exit 1
}
function ei {
  echo "[INFO] $PROG: $@" >&2;
}
function ed {
  [ -n "$VERBOSE" ] && echo "[DEBUG] $PROG: $@" >&2;
}
function ew {
  echo "[WARN] $PROG: $@" >&2;
}
export -f ee ei ed ew
###################################
# Defaults
###################################
export BATCH_SIZE=240
export N_EPOCHS=6
export N_NODES=1
export SAVE="model_qm9.pth"
export PPN=gpu
export AMP=true
export VERBOSE=
export DISTRIBUTED=
export LINK=auto
export NCCL_DEBUG=WARN

###################################
# Handle CLI arguments
###################################
function usage {
  echo """Create a Vagrant cluster for testing DeepOps deployment

Usage: $PROG training|interence [-h] [-v] [-d] [-A] [-D]
             [-P INT] [-N INT] [-E INT] [-B INT] [-S STR]

required arguments:
 training  Runs training benchmark
 inference Runs inference benchmark

optional Distributed arguments:
 -D STR ENABLE Distributed training over the following interfaces:
         - auto: NCCL chooses the device
         - ib: Force IB communication
         - ethernet: Disable IB communication
 -L STR Force intra-node communication over:
         - [auto]: NCCL chooses the device
         - nvlink: Communication goes through NVLINK
         - pcie: Communication goes through pcie
 -N INT Number of nodes [${N_NODES}]
 -P INT Number of processes per node [${PPN}]

optional Benchmarking arguments:
 -B INT Batch size [${BATCH_SIZE}]
 -E INT Number of training epochs [${N_EPOCHS}]
 -S STR Checkpoint save path [${SAVE}]
 -A     DISABLE automatic mixed precision

optional arguments:
 -v     Enable verbose logging
 -d     Enable debug logging
 -h     Print this help text""" >&2; exit 0
}

BENCHMARK=
while [ $OPTIND -le "$#" ]
do
  if getopts hdvAP:N:E:B:S:D:L: option
  then
    case $option
    in
      B) export BATCH_SIZE=${OPTARG};;
      E) export N_EPOCHS=${OPTARG};;
      N) export N_NODES=${OPTARG};;
      P) export PPN=${OPTARG};;
      S) export SAVE=${OPTARG};;
      L) export LINK=${OPTARG};;
      D) export DISTRIBUTED=${OPTARG};;
      A) export AMP=false;;
      v) export VERBOSE=1;;
      d) export NCCL_DEBUG=INFO;;
      :) echo -e "[ERROR] Missing an argument for ${OPTARG}\n" >&2; usage;;
      \?) echo -e "[ERROR] Illegal option ${OPTARG}\n" >&2; usage;;
      h) usage;;
    esac
  else
    if [ "${!OPTIND}" = "training" -o "${!OPTIND}" = "inference" ]; then
      BENCHMARK=${!OPTIND}
    else
      usage
    fi
    ((OPTIND++))
  fi
done

###################################
# Handle CLI arguments
###################################

# Handle distributed frabric
if [ -n "${DISTRIBUTED}" ]; then
  case $DISTRIBUTED; in
    auto) ei "Allowing NCCL to choose inter-node fabric";;
    ib) ei "Forcing NCCL to use infiniband interfaces"
      export NCCL_SOCKET_IFNAME=$({ which ifconfig &> /dev/null && ifconfig || ip a; } | grep -oE "(mlx|ib)[^: ]+:" | grep -oE "[^:]+" | tr "\n" "," | rev | cut -c2- | rev)
      ed "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}";;
    ethernet) ei "Forcing NCCL to use ethernet interfaces"
      export NCCL_SOCKET_IFNAME=$({ which ifconfig &> /dev/null && ifconfig || ip a; } | grep -oE "(eth|enp)[^: ]+:" | grep -oE "[^:]+" | tr "\n" "," | rev | cut -c2- | rev)
      export NCCL_IB_DISABLE=1
      ed "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}";;
    *) ee "${DISTRIBUTED} is not a valid interface option";;
  esac
  case $LINK; in
    auto) ei "Allowing NCCL to choose the intra-node fabric";;
    nvlink) ei "Forcing NCCL to use NVLink for intra-node communication"
      export NCCL_P2P_LEVEL=NVL
    pcie) ei "Forcing NCCL to use PCIe for intra-node communication"
      export NCCL_P2P_DISABLE=1
    *) ee "${LINK} is not a valid intra-node option";;
  esac
  ei "Using ${N_NODES} nodes and ${PPN} processes per node."
fi

###################################
# Run benchmarks
###################################
module=se3_transformer.runtime.${BENCHMARK}
common_flags="${module} --amp $AMP --batch_size $BATCH_SIZE --use_layer_norm --norm --task homo --seed 42 --benchmark"

if [ "${BENCHMARK}" == "inference" ]; then
  CMD="CUDA_VISIBLE_DEVICES=0 python -m ${common_flags}"
  ei "Running inference benchmark with:${CMD}"
else
  # Training
  training_flags="--epochs ${N_EPOCHS} --save_ckpt_path ${SAVE} --precompute_bases"
  if [ -n "${DISTRIBUTED}" ]; then
    # Distributed
    CMD="python -m torch.distributed.run --nnodes=${N_NODES} --nproc_per_node=${PPN} --max_restarts 0 --module ${common_flags} ${training_flags}"
    ei "Running distributed training benchmark with:${CMD}"
  else
    # Single-GPU
    CMD="CUDA_VISIBLE_DEVICES=0 python -m ${common_flags} ${training_flags}"
    ei "Running single-GPU training benchmark with:${CMD}"
  fi
fi

# RUN
$CMD
