setup:
	pip install torchfile==0.1.0
	cp config/solver.json.default config/solver.json
	mkdir -p pretrained
	wget -NP pretrained https://github.com/PavlosMelissinos/enet-keras/releases/download/v0.0.1-model/model-best.net
	PYTHONPATH=.:${PYTHONPATH} python src/models/from_torch.py

train:
	#LD_LIBRARY_PATH=/usr/local/cuda/lib64
	PYTHONPATH=.:${PYTHONPATH} python src/run.py --mode train --solver config/solver.json

predict:
	PYTHONPATH=.:${PYTHONPATH} python src/run.py --mode predict --solver config/solver.json

overfit:
	export LD_LIBRARY_PATH=/usr/local/cuda/lib64
	PYTHONPATH=.:${PYTHONPATH} python src/run.py --mode overfit --solver config/solver.json

test:
	#LD_LIBRARY_PATH=/usr/local/cuda/lib64
	# export CUDA_VISIBLE_DEVICES=1
	PYTHONPATH=.:${PYTHONPATH} python src/test.py
