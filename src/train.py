from __future__ import absolute_import, print_function

import json
import numpy as np
import os
import sys

from keras.callbacks import TensorBoard, ModelCheckpoint
from keras import backend as K

from data.data_loader import load_data
from data import mscoco as dataset
from models.enet import autoencoder, transfer_weights


def callbacks(log_dir, checkpoint_dir, model_name):
    tb = TensorBoard(log_dir=log_dir,
                     histogram_freq=1,
                     write_graph=True,
                     write_images=True)
    best_model = os.path.join(checkpoint_dir, '{}_best.h5'.format(model_name))
    save_best = ModelCheckpoint(
        best_model,
        monitor='val_loss',
        # verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode='auto',
        period=1)
    checkpoint_file = os.path.join(checkpoint_dir, 'weights.' + model_name + '.{epoch:02d}-{val_loss:.2f}.h5')
    checkpoints = ModelCheckpoint(
        filepath=checkpoint_file,
        monitor='val_loss',
        verbose=1,
        save_best_only=False,
        save_weights_only=False,
        mode='auto',
        period=1)

    # keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
    return [tb, save_best, checkpoints]


def train(solver, dataset_name):
    model_name = solver['model_name']

    print('Preparing to train on {} data...'.format(dataset_name))

    nb_epoch = solver['nb_epoch']
    batch_size = solver['batch_size']
    completed_epochs = solver['completed_epochs']
    skip = solver['skip']

    np.random.seed(1337) # for reproducibility

    dw = solver['dw']
    dh = solver['dh']
    data_shape = dw * dh
    nc = len(dataset.ids())  # categories + background

    autoenc, model_name = autoencoder(nc=nc, input_shape=(dw, dh))
    if 'h5file' in solver:
        h5file = solver['h5file']
        print('Loading model {}'.format(h5file))
        h5file, ext = os.path.splitext(h5file)
        autoenc.load_weights(h5file + ext)
    else:
        autoenc = transfer_weights(autoenc)

    if K.backend() == 'tensorflow':
        print('Tensorflow backend detected; Applying memory usage constraints')
        ss = K.tf.Session(config=K.tf.ConfigProto(gpu_options=K.tf.GPUOptions(allow_growth=True)))
        K.set_session(ss)
        ss.run(K.tf.global_variables_initializer())

    print('Done loading {} model!'.format(model_name))

    experiment_dir = os.path.join('models', dataset_name, model_name)
    log_dir = os.path.join(experiment_dir, 'logs')
    checkpoint_dir = os.path.join(experiment_dir, 'weights')

    train_gen = load_data(dataset,
                               data_dir=os.path.join('data', dataset_name),
                               batch_size=batch_size,
                               nc=nc,
                               target_hw=(dw, dh),
                               data_type='train2014',
                               shuffle=True)
    nb_train_samples = train_gen.next()  # first generator item is the count

    val_gen = load_data(dataset, 
                               data_dir=os.path.join('data', dataset_name),
                               batch_size=batch_size,
                               nc=nc,
                               target_hw=(dw, dh),
                               data_type='val2014',
                               sample_size=nb_train_samples//10)
    nb_val_samples = val_gen.next()  # first generator item is the count
    autoenc.fit_generator(
        generator=train_gen,
        samples_per_epoch=nb_train_samples,
        nb_epoch=nb_epoch,
        verbose=1,
        callbacks=callbacks(log_dir, checkpoint_dir, model_name),
        validation_data=val_gen,
        nb_val_samples=nb_val_samples,
        initial_epoch=completed_epochs
    )  # start from epoch e


if __name__ == '__main__':
    solver_json = 'config/solver.json'

    print('solver json: {}'.format(os.path.abspath(solver_json)))

    solver = json.load(open(solver_json))

    train(solver=solver, dataset_name='mscoco')
