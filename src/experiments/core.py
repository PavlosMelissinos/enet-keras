# coding=utf-8
from __future__ import absolute_import, division, print_function
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard

import numpy as np
import os

from src.data import utils as data_utils, datasets
from src.models import select_model
from src.models.objectives import w_categorical_crossentropy


class Experiment(object):
    def __init__(self, data, experiment, model, **kwargs):
        # set dataset attributes
        self.data_config = data
        self.DatasetClass = getattr(datasets, data['dataset_name'])

        # set basic model attributes
        self.model_config = model
        self.model_name = model['name']

        # set basic experiment attributes/directories
        self.experiment_config = experiment
        experiment_dir = os.path.join(
            self.experiment_config['root_dir'],
            self.data_config['dataset_name'],
            self.model_name
        )
        log_dir = os.path.join(experiment_dir, 'logs')
        checkpoint_dir = os.path.join(experiment_dir, 'weights')
        data_utils.ensure_dir(log_dir)
        data_utils.ensure_dir(checkpoint_dir)
        self.experiment_dir = experiment_dir
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir

        checkpoint_filename = 'weights.{}.{}.{}-{}.h5'.format(
            self.data_config['dataset_name'],
            self.model_name,
            '{epoch:02d}',
            '{val_loss:.2f}'
        )
        self.checkpoint_file = os.path.join(checkpoint_dir, checkpoint_filename)

    def callbacks(self):
        """
        :return:
        """
        # TODO: Add ReduceLROnPlateau callback
        cbs = []

        tb = TensorBoard(log_dir=self.log_dir,
                         write_graph=True,
                         write_images=True)
        cbs.append(tb)

        best_model_filename = self.model_name + '_best.h5'
        best_model = os.path.join(self.checkpoint_dir, best_model_filename)
        save_best = ModelCheckpoint(best_model, save_best_only=True)
        cbs.append(save_best)

        checkpoints = ModelCheckpoint(filepath=self.checkpoint_file, verbose=1)
        cbs.append(checkpoints)

        reduce_lr = ReduceLROnPlateau(patience=1, verbose=1)
        cbs.append(reduce_lr)
        return cbs

    def dataset(self, data_split='train'):
        """

        :param data_split: 'train', 'val' or 'test'
        :return:
        """
        valid_data_splits = ['train', 'val', 'test']
        if data_split not in valid_data_splits:
            errmsg = 'Invalid data split: {} instead of {}'.format(
                data_split,
                '/'.join(valid_data_splits)
            )
            raise ValueError(errmsg)

        kwargs = self.data_config
        supplementary_args = kwargs[data_split]
        kwargs.update(supplementary_args)

        return self.DatasetClass(**kwargs)

    def model(self):
        model = select_model(model_name=self.model_name)
        kwargs = self.model_config
        kwargs['nc'] = self.DatasetClass.num_classes()
        weights = self.DatasetClass.class_weights()
        kwargs['loss'] = [
            'categorical_crossentropy',
            # w_categorical_crossentropy(weights=weights)
        ]
        autoencoder, _ = model.build(**kwargs)
        if self.model_config['print_summary']:
            autoencoder.summary()
        try:
            h5file = self.model_config['h5file']
            print('Loading from file {}'.format(h5file))
            h5file, ext = os.path.splitext(h5file)
            autoencoder.load_weights(h5file + ext)
        except KeyError:
            autoencoder = model.transfer_weights(autoencoder)

        print('Done loading {} model!'.format(self.model_name))
        return autoencoder

    def run(self):
        dataset_name = self.data_config['dataset_name']
        print('Preparing to train on {} data...'.format(dataset_name))
        train_dataset = self.dataset(data_split='train')
        val_dataset = self.dataset(data_split='val')
        model = self.model()
        model.fit_generator(
            generator=train_dataset.flow(),
            steps_per_epoch=train_dataset.steps,
            epochs=self.experiment_config['epochs'],
            verbose=1,
            callbacks=self.callbacks(),
            validation_data=val_dataset.flow(),
            validation_steps=val_dataset.steps,
            initial_epoch=self.experiment_config['completed_epochs'],
        )


class CaptioningExperiment(Experiment):
    def __init__(self, data, model, experiment, **kwargs):
        super(CaptioningExperiment, self).__init__(data, model, experiment)
        max_seq_len = self.data_config['max_caption_length'] + 2
        self.model_config['max_token_length'] = max_seq_len

    def run(self):
        dataset_name = self.data_config['dataset_name']
        print('Preparing to train on {} data...'.format(dataset_name))
        train_dataset = self.dataset(data_split='train')
        val_dataset = self.dataset(data_split='val')
        self.model_config['vocab_size'] = train_dataset.vocab.size()
        model = self.model()
        model.fit_generator(
            generator=train_dataset.flow(),
            steps_per_epoch=train_dataset.steps,
            epochs=self.experiment_config['epochs'],
            verbose=1,
            callbacks=self.callbacks(),
            validation_data=val_dataset.flow(),
            validation_steps=val_dataset.steps,
            initial_epoch=self.experiment_config['completed_epochs'],
        )


class DryDatasetExperiment(Experiment):
    def __init__(self, data, model, experiment, **kwargs):
        super(DryDatasetExperiment, self).__init__(data, model, experiment, **kwargs)

    def run(self):
        dataset = self.dataset(data_split='val')
        for item in dataset.flow():
            for item_idx in range(dataset.config.batch_size):
                text = [dataset.vocab.decode(idx)
                        for idx in np.argmax(item[0]['text'][item_idx], axis=-1)]
                output = [dataset.vocab.decode(idx)
                          for idx in np.argmax(item[1]['output'][item_idx], axis=-1)]
                # print(' '.join(text))
                # print(' '.join(output))


class OverfittingCaptioningExperiment(CaptioningExperiment):
    def __init__(self, data, model, experiment, **kwargs):
        data['sample_size'] = 0.01
        super(OverfittingCaptioningExperiment, self).__init__(
            data, model, experiment, **kwargs
        )

    def run(self):
        dataset = self.dataset(data_split='val')
        for item in dataset.flow():
            for item_idx in range(dataset.config.batch_size):
                text = [dataset.vocab.decode(idx)
                        for idx in np.argmax(item[0]['text'][item_idx], axis=-1)]
                output = [dataset.vocab.decode(idx)
                          for idx in np.argmax(item[1]['output'][item_idx], axis=-1)]
                # print(' '.join(text))
                # print(' '.join(output))


class SemanticSegmentationExperiment(Experiment):
    def __init__(self, **kwargs):
        super(SemanticSegmentationExperiment, self).__init__(**kwargs)

    def model(self):
        model = select_model(model_name=self.model_name)
        kwargs = self.model_config
        kwargs['nc'] = self.DatasetClass.num_classes()
        weights = self.DatasetClass.class_weights()
        kwargs['loss'] = [
            'categorical_crossentropy',
            # w_categorical_crossentropy(weights=weights)
        ]
        autoencoder, _ = model.build(**kwargs)
        if self.model_config['print_summary']:
            autoencoder.summary()
        try:
            h5file = self.model_config['h5file']
            print('Loading from file {}'.format(h5file))
            h5file, ext = os.path.splitext(h5file)
            autoencoder.load_weights(h5file + ext)
        except KeyError:
            autoencoder = model.transfer_weights(autoencoder)

        print('Done loading {} model!'.format(self.model_name))
        return autoencoder
