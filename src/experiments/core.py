# coding=utf-8
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard

import os

from src.data import utils as data_utils, datasets
from src.models import select_model
from src.models.objectives import w_categorical_crossentropy


class Experiment(object):
    def __init__(self, solver):

        # setup experiment attributes
        self.epochs = solver['epochs']
        self._completed_epochs = solver['completed_epochs']

        # setup model attributes
        self.model_name = solver['model_name']
        self.model_config = solver

        # setup data attributes
        data_config = solver['data']
        data_config['h'] = solver['dh']
        data_config['w'] = solver['dw']
        self.data_config = data_config
        self.dataset_name = self.data_config['dataset_name']

        # setup experiment directories
        experiment_dir = os.path.join('models', self.dataset_name, self.model_name)
        log_dir = os.path.join(experiment_dir, 'logs')
        checkpoint_dir = os.path.join(experiment_dir, 'weights')
        data_utils.ensure_dir(log_dir)
        data_utils.ensure_dir(checkpoint_dir)
        self.experiment_dir = experiment_dir
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        checkpoint_filename = 'weights.' + self.model_name + '.{epoch:02d}-{val_loss:.2f}.h5'
        self.checkpoint_file = os.path.join(self.checkpoint_dir, checkpoint_filename)

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

        best_model = os.path.join(self.checkpoint_dir, '{}_best.h5'.format(self.model_name))
        save_best = ModelCheckpoint(best_model, save_best_only=True)
        cbs.append(save_best)

        checkpoints = ModelCheckpoint(filepath=self.checkpoint_file, verbose=1)
        cbs.append(checkpoints)

        reduce_lr = ReduceLROnPlateau(patience=1, verbose=1)
        cbs.append(reduce_lr)
        return cbs

    def model(self):
        model = select_model(model_name=self.model_name)
        dataset_name = self.dataset_name
        weights = datasets.load(dataset_name).class_weights
        loss = w_categorical_crossentropy(weights=weights)
        config = self.model_config
        nc = datasets.load(dataset_name).num_classes()
        w = config['dw']
        h = config['dh']
        autoencoder, model_name = model.build(nc=nc, w=w, h=h, loss=loss)
        if 'h5file' in self.model_config:
            h5file = self.model_config['h5file']
            print('Loading from file {}'.format(h5file))
            h5file, ext = os.path.splitext(h5file)
            autoencoder.load_weights(h5file + ext)
        else:
            autoencoder = model.transfer_weights(autoencoder)

        print('Done loading {} model!'.format(model_name))
        return autoencoder

    def dataset(self, data_split='train'):
        """

        :param data_split: 'train', 'val' or 'test'
        :return:
        """

        # transfer target dimensions to data configuration dictionary
        dataset_name = self.dataset_name

        assert data_split in ['train', 'val', 'test']

        data_config = self.data_config
        supplementary_data_config = data_config[data_split]
        data_config.update(supplementary_data_config)

        return datasets.load(dataset_name=dataset_name)(data_config)

    def run(self):
        print('Preparing to train on {} data...'.format(self.dataset_name))
        train_dataset = self.dataset(data_split='train')
        val_dataset = self.dataset(data_split='val')
        model = self.model()
        model.fit_generator(
            generator=train_dataset.flow(),
            steps_per_epoch=train_dataset.steps,
            epochs=self.epochs,
            verbose=1,
            callbacks=self.callbacks(),
            validation_data=val_dataset.flow(),
            validation_steps=val_dataset.steps,
            initial_epoch=self._completed_epochs,
        )
