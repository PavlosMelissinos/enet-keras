# coding=utf-8
from __future__ import absolute_import, division, print_function

from keras import backend as K
import json
import numpy as np
import os
import sys

from src.experiments.core import SemanticSegmentationExperiment as Experiment


if __name__ == '__main__':
    np.random.seed(1337)  # for reproducibility
    if K.backend() == 'tensorflow':
        print('Tensorflow backend detected; Applying memory usage constraints')
        config = K.tf.ConfigProto(gpu_options=K.tf.GPUOptions(allow_growth=True))
        ss = K.tf.Session(config=config)
        K.set_session(ss)
        ss.run(K.tf.global_variables_initializer())

    if len(sys.argv) > 1:
        solver_json = sys.argv[1]
    else:
        solver_json = 'config/solver.json'
    print('solver json: {}'.format(os.path.abspath(solver_json)))

    kwargs = json.load(open(solver_json))
    experiment = Experiment(**kwargs)
    experiment.run()
