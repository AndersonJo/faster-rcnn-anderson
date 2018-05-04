from argparse import ArgumentParser
from datetime import datetime

import cv2
import tensorflow as tf
import keras.backend as K
from keras.backend.tensorflow_backend import set_session
from tensorflow.python import debug as tf_debug

from frcnn.config import singleton_config, Config
from frcnn.classifier_trainer import ClassifierTrainer
from frcnn.fen import FeatureExtractionNetwork
from frcnn.rpn_trainer import RPNTrainer
from frcnn.classifier import ClassifierNetwork
from frcnn.rpn import RegionProposalNetwork
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
    fen = FeatureExtractionNetwork(config, input_shape=(None, None, 3))
    rpn = RegionProposalNetwork(fen, config)
    clf = ClassifierNetwork(rpn, config, class_mapping)

    for _ in range(50):
        # Train region proposal network
        now = datetime.now()
        batch_img, batch_cls, batch_regr, img_meta = rpn_trainer.next_batch()
        # print('batch_img:', batch_img.shape)
        # print('batch_cls:', batch_cls.shape)
        # print('batch_regr:', batch_regr.shape)
        # print('next batch 처리시간:', datetime.now() - now)

        rpn_cls_y, rpn_reg_y = rpn.model.predict_on_batch(batch_img)

        rpn_loss = rpn.model.train_on_batch(batch_img, [batch_cls, batch_regr])
        # print('loss_rpn:', loss_rpn)
        # print('rpn.model.train_on_batch 처리시간:', datetime.now() - now)

        cls_output, reg_output = rpn.model.predict_on_batch(batch_img)

        # print('cls_output:', cls_output.shape)
        # print('reg_output:', reg_output.shape)
        # print('rpn.model.predict_on_batch 처리시간:', datetime.now() - now)

        anchors, probs = clf.non_maximum_suppression(cls_output, reg_output)
        rois, cls_y, reg_y = clf_trainer.next_batch(anchors, img_meta)
        if rois is None:
            continue

        cls_pred, reg_pred = clf.model.predict_on_batch([batch_img, rois])

        clf_loss = clf.model.train_on_batch([batch_img, rois], [cls_y, reg_y])

        print('rpn_loss:', rpn_loss, 'clf_loss:', clf_loss, )

        # image = cv2.imread(datum['image_path'])
        # image = cv2.resize(image, (datum['rescaled_width'], datum['rescaled_height']))
        #
        # for i in range(nms_anchors.shape[0]):
        #     anc = nms_anchors[i] * 16
        #     cv2.rectangle(image, (anc[0], anc[1]), (anc[0] + 5, anc[1] + 5), (0, 0, 255))
        # cv2.imwrite(datum['filename'], image)


if __name__ == '__main__':
    config = singleton_config(parser)

    if parser.mode == 'train':
        train(config)
