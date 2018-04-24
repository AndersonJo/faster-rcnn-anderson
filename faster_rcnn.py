from argparse import ArgumentParser

from frcnn.config import singleton_config, Config
from frcnn.model import FasterRCNN
from frcnn.preprocessing import AnchorGenerator, singleton_anchor_thread_manager
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

    # Create Faster R-CNN Model
    frcnn = FasterRCNN(config.net_name, n_class=len(classes), rpn_depth=512)

    # Train region proposal network
    batch_img, batch_cls, batch_regr = anchor.next_batch()
    print('batch_img:', batch_img.shape)
    print('batch_cls:', batch_cls.shape)
    print('batch_regr:', batch_regr.shape)

    x = frcnn.rpn.train_on_batch(batch_img, batch_cls)
    print('x:', x)

    anchor_thread_mgr = singleton_anchor_thread_manager()
    anchor_thread_mgr.wait()


if __name__ == '__main__':
    config = singleton_config(parser)

    if parser.mode == 'train':
        train(config)
