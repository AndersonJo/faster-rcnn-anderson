import cv2
import os

from frcnn.preprocessing import cal_rescaled_size, rescale_image
from frcnn.voc import PascalVocData
from tests import DATASET_ROOT_PATH


def test_image_rescale():
    # Test Rescaled Size
    assert (600, 800) == cal_rescaled_size(300, 400, min_side=600)
    assert (600, 840) == cal_rescaled_size(50, 70, min_side=600)
    assert (2400, 600) == cal_rescaled_size(800, 200, min_side=600)
    assert (1024, 601) == cal_rescaled_size(1024, 601, min_side=600)
    assert (600, 600) == cal_rescaled_size(600, 600, min_side=600)
    assert (1000, 1000) == cal_rescaled_size(600, 600, min_side=1000)
    assert (1000, 2250) == cal_rescaled_size(20, 45, min_side=1000)

    # Test Rescaled Image
    vocdata = PascalVocData(DATASET_ROOT_PATH)
    train, test, classes = vocdata.load_data()
    dataset = train + test

    assert len(dataset) > 0
    for voc in dataset:
        # Check if the image exists
        assert os.path.exists(voc['image'])

        _img = cv2.imread(voc['image'])
        height, width, _ = _img.shape
        resized_width, resized_height = cal_rescaled_size(voc['width'], voc['height'], min_side=1000)
        resized_img = rescale_image(_img, resized_width, resized_height)

        # Check if the rescaled size is right
        assert voc['width'] == width
        assert voc['height'] == height
        assert (resized_height, resized_width, 3) == resized_img.shape
