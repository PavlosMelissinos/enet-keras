setup:
	pip install torchfile==0.1.0
	cp config/solver.json.default config/solver.json
	mkdir -p pretrained
	wget -NP pretrained https://github.com/PavlosMelissinos/enet-keras/releases/download/v0.0.1-model/model-best.net
	python src/models/from_torch.py

pycocotools:
	pip install Cython
	cd src/data/pycocotools && make

train:
	export PYTHONPATH=".":$PYTHONPATH
	#LD_LIBRARY_PATH=/usr/local/cuda/lib64
	python src/run.py --mode train --solver config/solver.json

predict:
	python src/run.py --mode predict --solver config/solver.json

overfit:
	export PYTHONPATH=".":$PYTHONPATH
	export LD_LIBRARY_PATH=/usr/local/cuda/lib64
	python src/run.py --mode overfit --solver config/solver.json

test:
	export PYTHONPATH=".":$PYTHONPATH
	#LD_LIBRARY_PATH=/usr/local/cuda/lib64
	# export CUDA_VISIBLE_DEVICES=1
	python src/test.py
