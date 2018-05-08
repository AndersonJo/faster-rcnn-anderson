from argparse import ArgumentParser
from datetime import datetime

import keras.backend as K
import numpy as np
import tensorflow as tf

from frcnn.classifier_trainer import ClassifierTrainer
from frcnn.config import singleton_config, Config
from frcnn.frcnn import FRCNN
from frcnn.rpn_trainer import RPNTrainer
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
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.8
tf_config.gpu_options.allow_growth = True

sess = tf.Session(config=tf_config)
# sess = tf_debug.LocalCLIDebugWrapperSession(sess)
K.set_session(sess)


# set_session(tf.Session(config=tf_config))


def train(config: Config):
    # Load data
    vocdata = PascalVocData(config.data_path)
    train, test, class_mapping = vocdata.load_data(limit_size=30, add_bg=True)

    # Load training tools
    rpn_trainer = RPNTrainer(train, shuffle=True, augment=True)
    clf_trainer = ClassifierTrainer(config, class_mapping)

    # Create Model
    frcnn = FRCNN(config, class_mapping)

    best_rpn_loss = np.inf
    best_clf_loss = np.inf

    for step in range(50):
        now = datetime.now()

        # Train Region Proposal Network
        batch_img, batch_cls, batch_regr, img_meta = rpn_trainer.next_batch()
        rpn_loss = frcnn.rpn_model.train_on_batch(batch_img, [batch_cls, batch_regr])

        # Train Classifier Network
        cls_output, reg_output = frcnn.rpn_model.predict_on_batch(batch_img)
        anchors, probs = frcnn.clf.non_maximum_suppression(cls_output, reg_output)
        # clf.debug_nms_images(anchors, img_meta)
        rois, cls_y, reg_y = clf_trainer.next_batch(anchors, img_meta)

        if rois is None:
            continue

        clf_loss = frcnn.clf.model.train_on_batch([batch_img, rois], [cls_y, reg_y])

        # cls_pred, reg_pred = clf.model.predict_on_batch([batch_img, rois])
        print('rpn_loss:', rpn_loss, 'clf_loss:', clf_loss)

        rpn_loss, clf_loss = rpn_loss[0], clf_loss[0]
        if rpn_loss < best_rpn_loss or clf_loss < best_clf_loss:
            best_rpn_loss = rpn_loss
            best_clf_loss = clf_loss
            filename = 'checkpoints/model_{0}_{1}_{2}.hdf5'.format(step, best_rpn_loss, best_clf_loss)
            frcnn.save(filename)
            print('Saved! best_rpn:{0} best_clf:{1}'.format(best_rpn_loss,
                                                            best_clf_loss))


if __name__ == '__main__':
    config = singleton_config(parser)

    if parser.mode == 'train':
        train(config)
