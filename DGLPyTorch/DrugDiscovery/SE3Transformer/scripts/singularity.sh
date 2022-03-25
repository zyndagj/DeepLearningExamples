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
export N_THREADS=1
export N_GPUS=$(nvidia-smi -L | grep GPU | wc -l)
export MIN_BYTES=8
export MAX_BYTES=128M
export ITERS=20
export WARM=5
export PARALLEL_INIT=0
export LAUNCH=
export VERBOSE=
export LINK=auto
export NCCL_DEBUG=WARN

###################################
# Handle CLI arguments
###################################
function usage {
  echo """Create a Vagrant cluster for testing DeepOps deployment

Usage: $PROG local|mpi [-h] [-v] [-d] [-t INT] [-g INT]
             [-b STR] [-e STR] [-n INT] [-w INT] [-P] [-L STR]

required arguments:
 local  Runs single-node performance benchmarks
 mpi    Runs multi-node performance benchmarks

optional arguments:
 -t INT Number of threads per process [${N_THREADS}]
 -g INT Number of GPUs per process [${N_GPUS}]
 -b STR Minimum number of bytes to scan [${MIN_BYTES}]
 -e STR Maximum number of bytes to scan [${MAX_BYTES}]
 -n INT Number of iterations [${ITERS}]
 -w INT Number of untimed warmup iterations [${ITERS}]

optional MPI-only arguments:
 -L STR Launch command
 -p INT Use threads to initialize NCCL [${PARALLEL_INIT}]

optional arguments:
 -v     Enable verbose logging
 -d     Enable debug logging
 -h     Print this help text""" >&2; exit 0
}

BENCHMARK=
while [ $OPTIND -le "$#" ]
do
  if getopts hdvp:L:w:n:e:b:g:t: option
  then
    case $option
    in
      t) export N_THREADS=${OPTARG};;
      g) export N_GPUS=${OPTARG};;
      b) export MIN_BYTES=${OPTARG};;
      e) export MAX_BYTES=${OPTARG};;
      n) export ITERS=${OPTARG};;
      w) export WARM=${OPTARG};;
      p) export PARALLEL_INIT=${OPTARG};;
      L) export LAUNCH="${OPTARG}";;
      v) export VERBOSE=1;;
      d) export NCCL_DEBUG=INFO;;
      :) echo -e "[ERROR] Missing an argument for ${OPTARG}\n" >&2; usage;;
      \?) echo -e "[ERROR] Illegal option ${OPTARG}\n" >&2; usage;;
      h) usage;;
    esac
  else
    if [ "${!OPTIND}" = "local" -o "${!OPTIND}" = "mpi" ]; then
      BENCHMARK=${!OPTIND}
    else
      usage
    fi
    ((OPTIND++))
  fi
done

###################################
# Run Benchmark
###################################

ARGS="-t ${N_THREADS} -g ${N_GPUS} -b ${MIN_BYTES} -e ${MAX_BYTES} -n ${ITERS} -w ${WARM} -p ${PARALLEL_INIT} -f 2"
GCMD="grep -oP '(?<=Avg bus bandwidth).*' | grep -oP '\d+\.\d+'"

# Handle distributed frabric
if [ -n "${DISTRIBUTED}" ]; then
  echo "Multi-node NCCL Tests"
  echo "test,interface,nodes,total_gpus,score"
  BINS=$(echo {all_reduce,all_gather,broadcast,reduce_scatter,reduce,alltoall,scatter,gather,sendrecv,hypercube}_perf_mpi)
  for B in $BINS; do
    CMD=$LAUNCH $B $ARGS
    # Auto
    unset NCCL_SOCKET_IFNAME
    unset NCCL_IB_DISABLE
    echo "$B,auto,${SLURM_NNODES},$(( ${SLURM_NTASKS}*${N_GPUS} )),$($CMD | $GCMD)"
    # IB
    export NCCL_SOCKET_IFNAME=$({ which ifconfig &> /dev/null && ifconfig || ip a; } | grep -oE "(mlx|ib)[^: ]+:" | grep -oE "[^:]+" | tr "\n" "," | rev | cut -c2- | rev)
    unset NCCL_IB_DISABLE
    echo "$B,ib,${SLURM_NNODES},$(( ${SLURM_NTASKS}*${N_GPUS} )),$($CMD | $GCMD)"
    # Ethernet
    export NCCL_SOCKET_IFNAME=$({ which ifconfig &> /dev/null && ifconfig || ip a; } | grep -oE "(eth|enp)[^: ]+:" | grep -oE "[^:]+" | tr "\n" "," | rev | cut -c2- | rev)
    export NCCL_IB_DISABLE=1
    echo "$B,eth,${SLURM_NNODES},$(( ${SLURM_NTASKS}*${N_GPUS} )),$($CMD | $GCMD)"
  done
else
  echo "Single-Node NCCL Tests"
  echo "test,interface,nodes,total_gpus,score"
  BINS=$(echo {all_reduce,all_gather,broadcast,reduce_scatter,reduce,alltoall,scatter,gather,sendrecv,hypercube}_perf)
  for B in $BINS; do
    CMD=$LAUNCH $B $ARGS
    # Auto
    unset NCCL_P2P_LEVEL
    unset NCCL_P2P_DISABLE
    echo "$B,auto,${SLURM_NNODES},$(( ${N_THREADS}*${N_GPUS} )),$($CMD | $GCMD)"
    # nvlink
    export NCCL_P2P_LEVEL=NVL
    unset NCCL_P2P_DISABLE
    echo "$B,nvlink,${SLURM_NNODES},$(( ${N_THREADS}*${N_GPUS} )),$($CMD | $GCMD)"
    # pcie
    unset  NCCL_P2P_LEVEL
    export NCCL_P2P_DISABLE=1
    echo "$B,pcie,${SLURM_NNODES},$(( ${N_THREADS}*${N_GPUS} )),$($CMD | $GCMD)"
  done
fi
