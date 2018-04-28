from argparse import ArgumentParser
from datetime import datetime

import cv2
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

from frcnn.config import singleton_config, Config
from frcnn.fen import FeatureExtractionNetwork
from frcnn.preprocessing import AnchorGenerator
from frcnn.roi import ROINetwork
from frcnn.rpn import RegionProposalNetwork
from frcnn.voc import PascalVocData

# Parse arguments

parser = ArgumentParser(description='Faster R-CNN')
parser.add_argument('--mode', default='train', type=str, help='train or test')
parser.add_argument('--data', default='/data/VOCdevkit', type=str, help='the path of VOC or COCO dataset')

# Base Model (Feature Extraction Network)ar
parser.add_argument('--net', default='vgg16', type=str, help='base network (vgg, resnet)')

# Reginon Proposal Network & Anchor
parser.add_argument('--rescale', default=True, type=bool, help='Rescale input image to lager one')

# Region Proposal Network Configuration

parser = parser.parse_args()

# Momory Limit
tf_config = tf.ConfigProto()
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.4
tf_config.gpu_options.allow_growth = True
set_session(tf.Session(config=tf_config))


def train(config: Config):
    # Load data
    vocdata = PascalVocData(config.data_path)
    train, test, classes = vocdata.load_data(limit_size=30)
    anchor = AnchorGenerator(train, shuffle=True, augment=True)

    # Create Model
    fen = FeatureExtractionNetwork(basenet=config.net_name, input_shape=(None, None, 3))
    rpn = RegionProposalNetwork(fen, config.anchor_scales, config.anchor_ratios, rpn_depth=512)
    roi = ROINetwork(rpn, n_class=len(classes))

    for _ in range(10):
        # Train region proposal network
        now = datetime.now()
        batch_img, batch_cls, batch_regr, datum = anchor.next_batch()
        # print('batch_img:', batch_img.shape)
        # print('batch_cls:', batch_cls.shape)
        # print('batch_regr:', batch_regr.shape)
        # print('next batch 처리시간:', datetime.now() - now)

        loss_rpn = rpn.model.train_on_batch(batch_img, [batch_cls, batch_regr])
        # print('loss_rpn:', loss_rpn)
        # print('rpn.model.train_on_batch 처리시간:', datetime.now() - now)

        cls_output, reg_output = rpn.model.predict_on_batch(batch_img)

        # print('cls_output:', cls_output.shape)
        # print('reg_output:', reg_output.shape)
        # print('rpn.model.predict_on_batch 처리시간:', datetime.now() - now)

        nms_anchors, nms_regrs = roi.rpn_to_roi(cls_output, reg_output)
        print('rpn_to_roi 까지 처리시간:', datetime.now() - now)

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
