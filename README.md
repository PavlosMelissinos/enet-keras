# ENet-keras

This is an implementation of [ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation](https://arxiv.org/abs/1606.02147), ported from [ENet-training](https://github.com/e-lab/ENet-training) ([lua-torch](https://github.com/torch/torch7)) to [keras](https://github.com/fchollet/keras).


## Requirements

[pycocotools](https://github.com/pdollar/coco)

OpenCV, Pillow and scikit-image - These are currently all listed as required. The reason is the project is undergoing a cleanup behind the scenes on this front. Currently OpenCV is used predominantly. However, it's a weird library and I'm attempting to replace it with something else. predict.py is used as a testbed.

hdf5

[tensorflow](https://www.tensorflow.org/) 

[keras](https://keras.io)


## Installation


### Anaconda/Miniconda 

```conda env create -f environment.yml```


### pip 

```pip install -r requirements.txt```


### pycocotools

Follow the instructions on the [repo](https://github.com/pdollar/coco) to install the MS-COCO API.


## Preparation

```git clone https://github.com/PavlosMelissinos/enet-keras.git```

### Set up pycocotools
Either set PYTHONPATH accordingly:

```export PYTHONPATH=path/to/MS-COCO/API/PythonAPI``` or

add a symbolic link to the pycocotools directory in the root of the project:

```ln -s path/to/MS-COCO/API/PythonAPI/pycocotools .```

### Prepare data

[Download ms-coco data](http://mscoco.org/dataset/#download):

```
cd data
./download_mscoco.sh
```

### Prepare pretrained ENet model

TODO

## Usage

### Predict

```python src/predict.py path/to/txt/file/containing/image/paths /path/to/h5/model /path/where/predictions/will/be/saved```

### Train on MS-COCO

TODO


## Remaining tasks

- [ ] Open new issue about available image processing libraries.
- [ ] download_mscoco.sh should extract the archives to their appropriate locations
- [ ] Upload pretrained model
- [ ] Finalize prediction.py
- [ ] Make data loader multithreaded
- [ ] Remove opencv from data loader
- [ ] Remove opencv from train.py
- [ ] Debug train.py
- [ ] Retrain ENet for rgb values
