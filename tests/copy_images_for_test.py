"""
원본 VOC Dataset의 이미지 데이터를 test 디렉토리로 옮겨줍니다.
test 데이터셋을 만들기 위해서 사용됩니다.

1. 먼저 사용하고자 하는 Annotations 파일들을 수작업으로 옮깁니다.
2. 이후 아래의 프로시져를 파이썬으로 실행시켜 자동으로 이미지를 복사합니다.

제약사항

1. 먼저 디렉토리가 수작업으로 만들어져있어야 합니다.
"""

import argparse
import os
import shutil
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))

from frcnn.voc import PascalVocData
from tests import DATASET_ROOT_PATH

parser = argparse.ArgumentParser()
parser.add_argument('--source', default='/data/VOCdevkit', type=str)
parser = parser.parse_args()


def copy_chunk(dataset: list):
    for voc in dataset:
        # print(voc['image'])
        os.path.basename(voc['image'])
        voc_name = os.path.basename(os.path.dirname(os.path.dirname(voc['image'])))
        filename = os.path.basename(voc['image'])

        source_path = os.path.join(parser.source, voc_name, 'JPEGImages', filename)
        dest_path = os.path.join(DATASET_ROOT_PATH, voc_name, 'JPEGImages', filename)

        shutil.copy(source_path, dest_path)


def main():
    pascal_voc = PascalVocData(DATASET_ROOT_PATH)
    train, test, classes = pascal_voc.load_data()
    copy_chunk(train)
    copy_chunk(test)


if __name__ == '__main__':
    main()
