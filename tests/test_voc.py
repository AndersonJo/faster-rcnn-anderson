import os
from typing import List

from frcnn.voc import PascalVocData

DATASET_ROOT_PATH = '/data/VOCdevkit/'


def test_voc_dataset_integrity():
    """
     - check if an image file exists
     - identify duplicate images in annotation data
    :return:
    """
    voc = PascalVocData(DATASET_ROOT_PATH)
    train, test, classes = voc.load_data()

    # check function
    def check(data: List[dict], visualize=True):
        if not hasattr(check, '_duplicate'):
            check._duplicate = list()

        for x in data:
            # Check whether an image file exists
            assert os.path.exists(x['image'])

            # Check duplicate
            assert x['image'] not in check._duplicate
            check._duplicate.append(x['image'])

            # if visualize:
            #     voc.visualize_img(x, 'files/haha{0}.jpg'.format(len(check._duplicate)))

    check(train)
    check(test)
