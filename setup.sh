#!/usr/bin/env bash
cp config/solver.json.default config/solver.json
mkdir pretrained
wget -P pretrained https://www.dropbox.com/sh/dywzk3gyb12hpe5/AABoUwqQGWvClUu27Z1EWeu9a/model-best.net
python src/models/from_torch.py
