import cv2
import os

from frcnn.tools import cal_rescaled_size, rescale_image, intersection, union, cal_iou, to_rescaled_coordinate
from frcnn.voc import PascalVocData
from tests import DATASET_ROOT_PATH


def test_image_rescale():
    """
    * test rescaled image size
    """
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
        assert os.path.exists(voc['image_path'])

        _img = cv2.imread(voc['image_path'])
        height, width, _ = _img.shape
        resized_width, resized_height = cal_rescaled_size(voc['width'], voc['height'], min_side=1000)
        resized_img = rescale_image(_img, resized_width, resized_height)

        # Check if the rescaled size is right
        assert voc['width'] == width
        assert voc['height'] == height
        assert (resized_height, resized_width, 3) == resized_img.shape


def test_converting_object_information_to_rescale_size():
    box1 = [2, 4, 6, 12]
    box2 = [3, 7, 7, 15]
    assert [250.0, 350.0, 750.0, 1050.0] == to_rescaled_coordinate(box1, 500, 700)
    assert [6.0, 28.0, 14.0, 60.0] == to_rescaled_coordinate(box2, 8, 32)


def test_intersection_over_union():
    box1 = [2, 4, 6, 12]
    box2 = [3, 7, 7, 15]

    assert 15 == intersection(box1, box2)
    assert 49 == union(box1, box2)
    assert 0.3061224427321951 == cal_iou(box1, box2)
