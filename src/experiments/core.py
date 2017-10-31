# coding=utf-8
from __future__ import absolute_import, division, print_function
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from matplotlib import pyplot as plt
# from keras.utils.vis_utils import plot_model

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
            # model_filename = self.model_name + '.png'
            # plot_model(autoencoder, to_file=model_filename, show_shapes=True)
        try:
            h5file = self.model_config['h5file']
            print('Loading from file {}'.format(h5file))
            h5file, ext = os.path.splitext(h5file)
            autoencoder.load_weights(h5file + ext)
        except KeyError:
            autoencoder = model.transfer_weights(autoencoder)
        return autoencoder

    def run(self):
        dataset_name = self.data_config['dataset_name']
        print('Preparing to train on {} data...'.format(dataset_name))
        train_dataset = self.dataset(data_split='train')
        val_dataset = self.dataset(data_split='val')
        model = self.model()

        print('Preparing to start training')
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


class InferenceExperiment(Experiment):
    def __init__(self, data, experiment, model, **kwargs):
        super(InferenceExperiment, self).__init__(data, experiment, model, **kwargs)

    def model(self):
        model_name = 'enet_unpooling'
        # model_name = 'enet_unpooling_weights_simple_setup'
        # model_name = 'enet_unpooling_no_weights'
        dataset_name = self.data_config['dataset_name']
        root_dir = 'experiments'
        pw = os.path.join(
            root_dir, dataset_name,
            model_name,
            'weights',
            # 'weights.enet_unpooling.02-2.59.h5'
            '{}_best.h5'.format(model_name)
        )

        # print(pw)

        nc = getattr(datasets, dataset_name).num_classes()
        self.model_config['nc'] = nc

        autoencoder = select_model(model_name=model_name)
        # segmenter, model_name = autoencoder.build(nc=nc, w=w, h=h)
        segmenter, model_name = autoencoder.build(**self.model_config)
        segmenter.load_weights(pw)
        return segmenter

    def run(self):
        model = self.model()
        dataset = self.dataset()
        for image_batch, target_batch in dataset.flow():
            image_batch = image_batch['image']
            target_batch = target_batch['output']
            for image, target in zip(image_batch, target_batch):
                output = model.predict(np.expand_dims(image, axis=0))[0]
                output = np.reshape(np.argmax(output, axis=-1), newshape=(512, 512))

                target = np.reshape(np.argmax(target, axis=-1), newshape=(512, 512))

                plt.rcParams["figure.figsize"] = [4 * 3, 4]

                fig = plt.figure()

                subplot1 = fig.add_subplot(131)
                subplot1.imshow(image.astype(np.uint8))
                subplot1.set_title('rgb image')
                subplot1.axis('off')

                subplot2 = fig.add_subplot(132)
                subplot2.imshow(output, cmap='gray')
                subplot2.set_title('Prediction')
                subplot2.axis('off')

                subplot3 = fig.add_subplot(133)
                masked = np.array(target)
                masked[target == 0] = 0
                subplot3.imshow(masked, cmap='gray')
                subplot3.set_title('Targets')
                subplot3.axis('off')

                fig.tight_layout()
                plt.show()


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
        super(DryDatasetExperiment, self).__init__(
            data=data,
            experiment=experiment,
            model=model,
            **kwargs
        )

    def split_label_channels(self, label):
        binary_masks = {}
        for i in range(label.shape[-1]):
            binary_mask = label[..., i]
            if not np.any(binary_mask > 0):
                continue
            binary_mask[binary_mask > 0] = 1
            binary_masks[i] = binary_mask.astype(np.uint8)
        return binary_masks

    def run(self):
        from matplotlib import pyplot as plt
        import sys

        np.random.seed(1337)  # for reproducibility

        dataset = self.dataset(data_split='val')

        for idx, item in enumerate(dataset.flow()):
            img, lbl = item[0]['image'].astype(np.uint8), item[1]['output']
            batch_size = img.shape[0]
            h = img.shape[1]
            w = img.shape[2]
            nc = lbl.shape[-1]
            lbl = np.reshape(lbl, (batch_size, h, w, nc))
            # batch_size = dataset.config.batch_size
            for batch_index in range(batch_size):
                binary_masks = self.split_label_channels(lbl[batch_index, ...])
                img_item = img[batch_index, ...]
                for class_idx, binary_mask in binary_masks.items():
                    # class_name = dataset.CATEGORIES[dataset.IDS[class_idx]]
                    class_name = dataset.CATEGORIES[class_idx]

                    plt.rcParams["figure.figsize"] = [4 * 3, 4]

                    fig = plt.figure()

                    subplot1 = fig.add_subplot(131)
                    subplot1.imshow(img_item)
                    subplot1.set_title('rgb image')
                    subplot1.axis('off')

                    subplot2 = fig.add_subplot(132)
                    subplot2.imshow(binary_mask, cmap='gray')
                    subplot2.set_title('{} binary mask'.format(class_name))
                    subplot2.axis('off')

                    subplot3 = fig.add_subplot(133)
                    masked = np.array(img_item)
                    masked[binary_mask == 0] = 0
                    subplot3.imshow(masked)
                    subplot3.set_title('{} label'.format(class_name))
                    subplot3.axis('off')

                    fig.tight_layout()
                    plt.show()
            # shapes.append(img.shape)
                item_idx = batch_size * idx + batch_index + 1
                print('Processed {} items: ({})'.format(item_idx, type(item)),
                      end='\r')
            sys.stdout.flush()


class DryDatasetCaptioningExperiment(CaptioningExperiment):
    def __init__(self, data, model, experiment, **kwargs):
        super(DryDatasetCaptioningExperiment, self).__init__(
            data=data,
            experiment=experiment,
            model=model,
            **kwargs
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


class OverfittingExperiment(Experiment):
    def __init__(self, data, model, experiment, **kwargs):
        data['sample_size'] = 100
        experiment['epochs'] = 50
        super(OverfittingExperiment, self).__init__(
            data=data,
            experiment=experiment,
            model=model,
            **kwargs
        )

    def run(self):
        dataset = self.dataset(data_split='train')
        model = self.model()
        model.fit_generator(
            generator=dataset.flow(),
            steps_per_epoch=dataset.steps,
            epochs=self.experiment_config['epochs'],
            verbose=1,
            # validation_data=dataset.flow(),
            # validation_steps=dataset.steps,
            initial_epoch=self.experiment_config['completed_epochs'],
        )
        for inputs, outputs in dataset.flow(single_pass=True):
            predictions = model.predict(inputs)
            for image, prediction, output in zip(inputs['image'], predictions, outputs['output']):
                # class_name = dataset.CATEGORIES[class_idx]

                plt.rcParams["figure.figsize"] = [4 * 3, 4]

                fig = plt.figure()

                subplot1 = fig.add_subplot(131)
                subplot1.imshow(image.astype(np.uint8))
                subplot1.set_title('rgb image')
                subplot1.axis('off')

                subplot2 = fig.add_subplot(132)
                # newshape = (image.shape[:2]) + (prediction.shape[-1],)
                prediction = np.argmax(prediction, axis=-1)
                newshape = image.shape[:2]
                prediction = np.reshape(prediction, newshape)
                subplot2.imshow(prediction, cmap='gray')
                # subplot2.set_title('{} binary mask'.format(class_name))
                subplot2.set_title('predicted output')
                subplot2.axis('off')

                subplot3 = fig.add_subplot(133)
                # newshape = (image.shape[:2]) + (output.shape[-1],)
                output = np.argmax(output, axis=-1)
                newshape = image.shape[:2]
                output = np.reshape(output, newshape)
                masked = np.array(output)
                masked[output == 0] = 0
                subplot3.imshow(masked, cmap='gray')
                # subplot3.set_title('{} label'.format(class_name))
                subplot3.set_title('ground truth labels')
                subplot3.axis('off')

                fig.tight_layout()
                plt.show()
        print('End of overfitting experiment')


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
