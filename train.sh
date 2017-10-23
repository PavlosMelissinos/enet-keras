#!/usr/bin/env bash
export PYTHONPATH=".":$PYTHONPATH
#LD_LIBRARY_PATH=/usr/local/cuda/lib64

python src/run.py --mode train --solver config/solver.json
