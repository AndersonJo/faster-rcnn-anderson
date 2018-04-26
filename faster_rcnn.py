from argparse import ArgumentParser

from frcnn.config import singleton_config, Config
from frcnn.model import FeatureExtractionNetwork, RegionProposalNetwork
from frcnn.preprocessing import AnchorGenerator, singleton_anchor_thread_manager
from frcnn.roi import ROINetwork
from frcnn.voc import PascalVocData

# Parse arguments

parser = ArgumentParser(description='Faster R-CNN')
parser.add_argument('--mode', default='train', type=str, help='train or test')
parser.add_argument('--data', default='/data/VOCdevkit', type=str, help='the path of VOC or COCO dataset')

# Base Model (Feature Extraction Network)ar
parser.add_argument('--net', default='vgg16', type=str, help='base network (vgg, resnet)')

# Reginon Proposal Network & Anchor
parser.add_argument('--thread', default=1, type=int, help='the number of threads for rpn target data')
parser.add_argument('--rescale', default=True, type=bool, help='Rescale input image to lager one')

# Region Proposal Network Configuration

parser = parser.parse_args()


def train(config: Config):
    # Load data
    vocdata = PascalVocData(config.data_path)
    train, test, classes = vocdata.load_data(limit_size=30)
    anchor = AnchorGenerator(train, batch=32)

    # Create Model
    fen = FeatureExtractionNetwork(basenet=config.net_name, input_shape=(None, None, 3))
    rpn = RegionProposalNetwork(fen, config.anchor_scales, config.anchor_ratios, rpn_depth=512)
    roi = ROINetwork(rpn, n_class=len(classes))

    # Train region proposal network
    batch_img, batch_cls, batch_regr = anchor.next_batch()
    print('batch_img:', batch_img.shape)
    print('batch_cls:', batch_cls.shape)
    print('batch_regr:', batch_regr.shape)

    loss_rpn = rpn.model.train_on_batch(batch_img, [batch_cls, batch_regr])
    print('loss_rpn:', loss_rpn)

    cls_output, reg_output = rpn.model.predict_on_batch(batch_img)

    print('cls_output:', cls_output.shape)
    print('reg_output:', reg_output.shape)

    nms_anchors, nms_regrs = roi.to_roi(cls_output, reg_output)

    import ipdb
    ipdb.set_trace()
    anchor_thread_mgr = singleton_anchor_thread_manager()
    anchor_thread_mgr.wait()


if __name__ == '__main__':
    config = singleton_config(parser)

    if parser.mode == 'train':
        train(config)
