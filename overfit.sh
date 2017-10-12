#!/usr/bin/env bash
export PYTHONPATH=".":$PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64

python src/run.py overfit config/solver.json
