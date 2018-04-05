from argparse import ArgumentParser

# Parse arguments
parser = ArgumentParser(description='Faster R-CNN')
parser.add_argument('--net', default='vgg16', type=str, help='base network (vgg, resnet)')
parser.add_argument('--data-path', default='/data/VOCdevkit')
parser = parser.parse_args()


def main():
    print(parser.data_path)


if __name__ == '__main__':
    main()
