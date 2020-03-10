# ENet-keras

[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](https://github.com/PavlosMelissinos/enet-keras/blob/master/LICENSE)
![](https://reposs.herokuapp.com/?path=PavlosMelissinos/enet-keras&style=flat&color=red)
[![Read the Docs](https://img.shields.io/readthedocs/pip.svg)]()

This is an implementation of [ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation](https://arxiv.org/abs/1606.02147), ported from [ENet-training](https://github.com/e-lab/ENet-training) ([lua-torch](https://github.com/torch/torch7)) to [keras](https://github.com/fchollet/keras).


## Installation

### Setup environment

#### Dependencies

On poetry: `poetry update`

On Anaconda/miniconda: `conda env create -f environment.yml`

On pip: `pip install -r requirements.txt`

#### One-time dependencies

`torchfile` in order to convert the torch model to a keras one.

### Get code

```
git clone https://github.com/PavlosMelissinos/enet-keras.git
cd enet-keras
```

### Set up data/model

```
make setup
```

The setup script only sets up some directories and converts the model to an appropriate format.

MSCOCO is only downloaded on demand. 

## Usage

### Train on MS-COCO

`make train`

## Remaining tasks

- [ ] Clean up code
  - [ ] Remove hardcoded paths
  - [ ] Add documentation everywhere
- [ ] Test code
  - [ ] Add tests
- [ ] Fix performance (mostly preprocessing bottleneck)
  - [ ] Remove unnecessary computations in data preprocessing
  - [ ] Index dataset category internals. Dataset categories have fields with one-to-one correspondence like id, category_id, palette, categories. This seems like perfect table structure. Might be too much though.
  - [ ] (Optionally) Make data loader multithreaded (no idea how to approach this one, multithreadedness is handled by keras though)
- [ ] Enhance reproducibility/usability
  - [x] Upload pretrained model
  - [ ] Finalize predict.py
    - [x] Test whether it works after latest changes
    - [ ] Modify predict.py to load a single image or from a file. There's no point in loading images from the validation set.
- [ ] Fix bugs
  - [ ] Investigate reason for bad results, see [#11](https://github.com/PavlosMelissinos/enet-keras/issues/11)
  - [ ] Fix MSCOCOReduced, [also see #9](https://github.com/PavlosMelissinos/enet-keras/issues/9)
  - [ ] ?????
