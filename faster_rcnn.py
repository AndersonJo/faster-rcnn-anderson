from argparse import ArgumentParser

# Parse arguments
from frcnn.preprocessing import get_anchor
from frcnn.voc import PascalVocData

parser = ArgumentParser(description='Faster R-CNN')
parser.add_argument('--mode', default='train', type=str, help='train or test')
parser.add_argument('--net', default='vgg16', type=str, help='base network (vgg, resnet)')
parser.add_argument('--data-path', default='/data/VOCdevkit')
parser = parser.parse_args()


def load_dataset():
    vocdata = PascalVocData(parser.data_path)
    train, test, classes = vocdata.load_data(limit_size=30)

    train = get_anchor(train)


def train():
    load_dataset()


if __name__ == '__main__':
    if parser.mode == 'train':
        train()
