from datetime import datetime

import cv2

from frcnn.config import singleton_config
from frcnn.fen import FeatureExtractionNetwork
from frcnn.nms import non_max_suppression
import numpy as np

from frcnn.rpn_trainer import RPNTrainer
from frcnn.detector import DetectionNetwork
from frcnn.rpn import RegionProposalNetwork
from frcnn.voc import PascalVocData
from tests import DATASET_ROOT_PATH


def test_non_maximum_suppress():
    """
    The code is from https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
    The test code cannot automatically determine if the nms code works properly
    """
    images = [
        ("nms/zombies_01.jpg", np.array([
            (180, 72, 244, 136),
            (186, 78, 250, 142),
            (186, 72, 250, 136)])),
        ("nms/zombies_02.jpg", np.array([
            (504, 306, 568, 370),
            (217, 150, 395, 328)])),
        ("nms/sarah4.jpg", np.array([
            (66, 100, 244, 278),
            (83, 100, 261, 278),
            (66, 117, 244, 295),
            (83, 117, 261, 295),
            (66, 133, 244, 311),
            (83, 133, 261, 311)]))]

    for i, (img_path, bboxes) in enumerate(images):
        image = cv2.imread(img_path)
        for bbox in bboxes:
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)

        picks, _ = non_max_suppression(bboxes, overlap_threshold=0.3)

        for pick in picks:
            cv2.rectangle(image, (pick[0], pick[1]), (pick[2], pick[3]), (0, 255, 0), 2)
        cv2.imwrite('nms/pick_{0}.jpg'.format(i), image)


def test_non_maximum_suppress_with_probabilities():
    """
    The code is from https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
    The test code cannot automatically determine if the nms code works properly
    """
    images = [
        ("nms/zombies_01.jpg", np.array([
            (180, 72, 244, 136),
            (186, 78, 250, 142),
            (186, 72, 250, 136)])),
        ("nms/zombies_02.jpg", np.array([
            (504, 306, 568, 370),
            (217, 150, 395, 328)])),
        ("nms/sarah4.jpg", np.array([
            (66, 100, 244, 278),
            (83, 100, 261, 278),
            (66, 117, 244, 295),
            (83, 117, 261, 295),
            (66, 133, 244, 311),
            (83, 133, 261, 311)]))]

    probabilities = [[0.2, 0.5, 0.3], [0.7, 0.3], [0.1, 0.1, 0.5, 0.1, 0.1, 0.1]]

    for i, (img_path, bboxes) in enumerate(images):
        prob = np.array(probabilities[i])

        image = cv2.imread(img_path)
        for bbox in bboxes:
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)

        picks, probs = non_max_suppression(bboxes, prob, overlap_threshold=0.3)

        for pick in picks:
            cv2.rectangle(image, (pick[0], pick[1]), (pick[2], pick[3]), (0, 255, 0), 2)
        cv2.imwrite('nms/prob_{0}.jpg'.format(i), image)


def _test_nms():
    """
    performance test
    """

    # Get config
    config = singleton_config()

    # Get Data
    vocdata = PascalVocData(DATASET_ROOT_PATH)
    train, test, classes = vocdata.load_data(limit_size=30)
    anchor = RPNTrainer(train, batch=6)

    # Create Model
    fen = FeatureExtractionNetwork(basenet='vgg16', input_shape=(None, None, 3))
    rpn = RegionProposalNetwork(fen, config.anchor_scales, config.anchor_ratios, rpn_depth=512)
    roi = DetectionNetwork(rpn, n_class=len(classes))

    # Predict
    now = datetime.now()
    for i in range(100):
        batch_image, cls_target, reg_target, datum = anchor.next_batch()
        cls_output, reg_output = rpn.model.predict_on_batch(batch_image)
        nms_anchors, nms_regrs = roi.rpn_to_roi(cls_output, reg_output)

    print('최종:', datetime.now() - now)
