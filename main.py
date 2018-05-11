from argparse import ArgumentParser
from datetime import datetime

import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.utils import Progbar

from frcnn.classifier_trainer import ClassifierTrainer
from frcnn.config import singleton_config, Config
from frcnn.frcnn import FRCNN
from frcnn.rpn_trainer import RPNTrainer, RPNDataProcessor
from frcnn.voc import PascalVocData

# Parser:: Basic Arguments
parser = ArgumentParser(description='Faster R-CNN')
parser.add_argument('--mode', default='train', type=str, help='train or test')
parser.add_argument('--data', default='/data/VOCdevkit', type=str, help='the path of VOC or COCO dataset')

# Parser:: Base Model (Feature Extraction Network)
parser.add_argument('--net', default='vgg16', type=str, help='base network (vgg, resnet)')

# Parser:: Reginon Proposal Network & Anchor
parser.add_argument('--rescale', default=True, type=bool, help='Rescale input image to lager one')
parser = parser.parse_args()

# Momory Limit & Debugging
tf_config = tf.ConfigProto()
tf_config.gpu_options.per_process_gpu_memory_fraction = 1.
tf_config.gpu_options.allow_growth = True

sess = tf.Session(config=tf_config)
# sess = tf_debug.LocalCLIDebugWrapperSession(sess)
K.set_session(sess)


# set_session(tf.Session(config=tf_config))


def train_voc(config: Config, train: list, class_mapping: dict):
    # Load training tools
    rpn_trainer = RPNTrainer(train, shuffle=True, augment=True)
    clf_trainer = ClassifierTrainer(config, class_mapping)

    # Create Model
    frcnn = FRCNN(config, class_mapping, train=True)

    # Progress Bar
    progbar = Progbar(len(train))

    best_loss = np.inf
    for step in range(len(train)):
        batch_images, batch_cls, batch_regr, img_meta = rpn_trainer.next_batch()

        # Train Region Proposal Network
        rpn_loss = frcnn.rpn_model.train_on_batch(batch_images, [batch_cls, batch_regr])

        # Train Classifier Network
        rpn_cls, rpn_reg = frcnn.rpn_model.predict_on_batch(batch_images)
        anchors, probs = frcnn.generate_anchors(rpn_cls, rpn_reg)
        # clf.debug_nms_images(anchors, img_meta)
        rois, cls_y, reg_y = clf_trainer.next_batch(anchors, img_meta)

        if rois is None:
            continue

        clf_loss = frcnn.clf_model.train_on_batch([batch_images, rois], [cls_y, reg_y])

        # cls_pred, reg_pred = clf.model.predict_on_batch([batch_img, rois])

        # Update Visualization
        total_loss = rpn_loss[0] + clf_loss[0]

        _saved = False
        if total_loss < best_loss and step > 1000:
            best_loss = total_loss
            filename = 'checkpoints/model_{0}_{1:.4}.hdf5'.format(step, round(total_loss, 4))
            frcnn.save(filename)
            _saved = True

            # print('Saved! best_rpn:{0} best_clf:{1}'.format(rpn_loss, clf_loss))

        progbar.update(step, [('rpn', rpn_loss[0]),
                              ('clf', clf_loss[0]),
                              ('clf_c', clf_loss[1]),
                              ('clf_r', clf_loss[2])])


def test_voc(config: Config, test: list, class_mapping: dict):
    # Load data tools
    rpn_data = RPNDataProcessor(test, shuffle=False, augment=False)

    # Create Model
    frcnn = FRCNN(config, class_mapping)
    frcnn.load('checkpoints/model_33758_0.11919999867677689.hdf5')

    # Inference
    for step in range(rpn_data.count()):
        batch_image, img_meta = rpn_data.next_batch()

        rpn_cls, rpn_reg, f_maps = frcnn.rpn_model.predict_on_batch(batch_image)
        anchors, probs = frcnn.generate_anchors(rpn_cls, rpn_reg)

        cls_ys = list()
        reg_ys = list()
        for rois in frcnn.iter_rois(anchors):
            cls_y, reg_y = frcnn.clf_model.predict_on_batch([batch_image, rois])
            cls_ys.append(cls_y)
            reg_ys.append(reg_y)

        import ipdb
        ipdb.set_trace()
        #     print(cls_y.shape, reg_y.shape)
        # print(step)


def main(config: Config):
    # Load data
    vocdata = PascalVocData(config.data_path)
    train, test, class_mapping = vocdata.load_data(limit_size=30, add_bg=True)

    if parser.mode == 'train':
        train_voc(config, train, class_mapping)
    elif parser.mode == 'test':
        test_voc(config, test, class_mapping)


if __name__ == '__main__':
    config = singleton_config(parser)
    main(config)
