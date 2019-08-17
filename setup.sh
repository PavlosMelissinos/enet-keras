#!/usr/bin/env bash
cp config/solver.json.default config/solver.json
mkdir pretrained
wget -P pretrained https://github.com/PavlosMelissinos/enet-keras/releases/download/v0.0.1-model/model-best.net
python src/models/from_torch.py
