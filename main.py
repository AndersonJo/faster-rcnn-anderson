import os
from datetime import datetime

from frcnn.logging import get_logger

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from argparse import ArgumentParser

import cv2
import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.utils import Progbar

from frcnn.classifier_trainer import ClassifierTrainer
from frcnn.config import singleton_config, Config
from frcnn.frcnn import FRCNN
from frcnn.nms import non_max_suppression
from frcnn.rpn_trainer import RPNTrainer, RPNDataProcessor
from frcnn.voc import PascalVocData

# Logger
logger = get_logger(__name__)

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
    inv_class_mapping = {v: k for k, v in class_mapping.items()}

    # Load training tools
    rpn_trainer = RPNTrainer(train, shuffle=True, augment=True)
    clf_trainer = ClassifierTrainer(config, class_mapping)

    # Create Model
    frcnn = FRCNN(config, class_mapping, train=True)
    frcnn.load_latest()

    # Progress Bar
    progbar = Progbar(len(train))

    # Parameters
    best_loss = np.inf
    global_step = 0

    for epoch in range(100):
        for step in range(len(train)):
            batch_image, original_image, batch_cls, batch_regr, meta = rpn_trainer.next_batch()

            # Train Region Proposal Network
            rpn_loss = frcnn.rpn_model.train_on_batch(batch_image, [batch_cls, batch_regr])

            # Train Classifier Network
            rpn_cls, rpn_reg = frcnn.rpn_model.predict_on_batch(batch_image)

            anchors, probs = frcnn.generate_anchors(rpn_cls, rpn_reg)
            # clf.debug_nms_images(anchors, img_meta)
            rois, cls_y, reg_y, best_ious = clf_trainer.next_batch(anchors, meta)

            if rois is None:
                continue

            clf_loss = frcnn.clf_model.train_on_batch([batch_image, rois], [cls_y, reg_y])
            # cls_pred, reg_pred = clf.model.predict_on_batch([batch_img, rois])

            # Update Visualization
            total_loss = rpn_loss[0] + clf_loss[0]

            if total_loss < best_loss and global_step > 1000:
                best_loss = total_loss
                filename = 'checkpoints/model_{0}_{1:.4}.hdf5'.format(step, round(total_loss, 4))
                frcnn.save(filename)

            # Progress Bar
            global_step += 1
            progbar.update(step, [('rpn', rpn_loss[0]),
                                  ('clf', clf_loss[0]),

                                  ('best_iou', len(best_ious)),
                                  ])

            print()

            _answer_class = [obj[0] for obj in meta['objects']]
            if len(set(_answer_class)) >= 2:
                _pred_class = [inv_class_mapping[idx] for idx in np.argmax(cls_y, axis=2).tolist()[0]]
                _pred_class = list(filter(lambda x: x != 'bg', _pred_class))

                print()
                print('predict:', _pred_class)
                print('answer:', _answer_class)
                print()

                # cls_indices, gta_regs = frcnn.clf_predict(batch_image, anchors, img_meta=meta)
                # gta_regs, cls_indices = non_max_suppression(gta_regs, cls_indices, overlap_threshold=0.5)
                # if gta_regs is not None:
                #     visualize(original_image, meta, gta_regs)

            # Visualize
            # cls_indices, gta_regs = frcnn.clf_predict(batch_image, anchors, img_meta=meta)
            # gta_regs, cls_indices = non_max_suppression(gta_regs, cls_indices, overlap_threshold=0.5)
            # if gta_regs is not None:
            #     visualize(original_image, meta, gta_regs)


def test_voc(config: Config, test: list, class_mapping: dict):
    class_mapping_inv = {v: k for k, v in class_mapping.items()}

    # Load data tools
    rpn_data = RPNDataProcessor(test, shuffle=False, augment=False)

    # Create Model
    frcnn = FRCNN(config, class_mapping)
    frcnn.load('checkpoints/model_33758_0.11919999867677689.hdf5')

    # Inference
    for step in range(rpn_data.count()):
        batch_image, original_image, meta = rpn_data.next_batch()

        rpn_cls, rpn_reg, f_maps = frcnn.rpn_model.predict_on_batch(batch_image)
        anchors, probs = frcnn.generate_anchors(rpn_cls, rpn_reg)

        cls_indices, gta_regs = frcnn.clf_predict(batch_image, anchors, img_meta=meta)
        gta_regs, cls_indices = non_max_suppression(gta_regs, cls_indices, overlap_threshold=0.5)
        if gta_regs is not None:
            visualize(original_image, meta, gta_regs)


def visualize(image, meta, gta_regs: np.ndarray):
    for reg in gta_regs:
        rescaled_ratio = meta['rescaled_ratio']
        min_x, min_y, max_x, max_y = (reg // rescaled_ratio).astype(np.uint8).tolist()
        cv2.rectangle(image, (min_x, min_y), (max_x, max_y), (0, 0, 255))

    cv2.imwrite('temp/{0}'.format(meta['filename']), image)


def main(config: Config):
    # Load data
    logger.info('loading data')
    vocdata = PascalVocData(config.data_path)
    train, test, class_mapping = vocdata.load_data(limit_size=30, add_bg=True)

    if parser.mode == 'train':
        train_voc(config, train, class_mapping)
    elif parser.mode == 'test':
        test_voc(config, test, class_mapping)


if __name__ == '__main__':
    config = singleton_config(parser)
    main(config)
