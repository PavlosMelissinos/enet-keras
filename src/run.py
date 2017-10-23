# coding=utf-8
from __future__ import absolute_import, division, print_function

from keras import backend as K
import argparse
import json
import numpy as np

from src import experiments


def run(mode, solver_file):
    print('solver json: {}'.format(solver_file))
    kwargs = json.load(open(solver_file))

    if mode == 'train':
        experiment = experiments.core.SemanticSegmentationExperiment(**kwargs)
    elif mode == 'predict':
        experiment = experiments.core.InferenceExperiment(**kwargs)
    elif mode == 'overfit':
        experiment = experiments.core.OverfittingExperiment(**kwargs)
    elif mode == 'data':
        experiment = experiments.core.DryDatasetExperiment(**kwargs)
    else:
        errmsg = 'This script only supports train/overfit/data at the moment'
        raise NotImplementedError(errmsg)

    experiment.run()


if __name__ == '__main__':
    desc = '''Script that loads configuration files and runs experiments.'''
    parser = argparse.ArgumentParser(
        description=desc,
        epilog='')
    parser.add_argument('--mode',
                        nargs='?',
                        type=str,
                        default='predict',
                        help='One of train/overfit/data')
    parser.add_argument('--solver',
                        nargs='?',
                        type=str,
                        default='config/solver.json',
                        help='')
    args = parser.parse_args()

    np.random.seed(1337)  # for reproducibility
    if K.backend() == 'tensorflow':
        print('Tensorflow backend detected; Applying memory usage constraints')
        config = K.tf.ConfigProto(gpu_options=K.tf.GPUOptions(allow_growth=True))
        ss = K.tf.Session(config=config)
        K.set_session(ss)
        ss.run(K.tf.global_variables_initializer())

    run(mode=args.mode, solver_file=args.solver)
