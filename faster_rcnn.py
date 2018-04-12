from argparse import ArgumentParser

# Parse arguments
from frcnn.preprocessing import Anchor, AnchorThreadManager
from frcnn.config import singleton_config, Config
from frcnn.voc import PascalVocData

parser = ArgumentParser(description='Faster R-CNN')
parser.add_argument('--mode', default='train', type=str, help='train or test')
parser.add_argument('--net', default='vgg16', type=str, help='base network (vgg, resnet)')
parser.add_argument('--data-path', default='/data/VOCdevkit')
parser.add_argument('--thread', default=1, type=int, help='the number of threads for rpn target data')

# Region Proposal Network Configuration
parser = parser.parse_args()


def load_dataset(config: Config):
    vocdata = PascalVocData(parser.data_path)
    train, test, classes = vocdata.load_data(limit_size=30)

    anchor = Anchor(train)
    anchors = anchor.next_batch()


def train(config: Config):
    anchor_thread_mgr = AnchorThreadManager(config.n_thread)
    anchor_thread_mgr.create_anchor_threads()

    load_dataset(config)


if __name__ == '__main__':
    config = singleton_config(parser)

    if parser.mode == 'train':
        train(config)
