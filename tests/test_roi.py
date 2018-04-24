import cv2

from frcnn.roi import non_max_suppression_fast
import numpy as np


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

        picks = non_max_suppression_fast(bboxes, 0.3)
        break

        for pick in picks:
            cv2.rectangle(image, (pick[0], pick[1]), (pick[2], pick[3]), (0, 255, 0), 2)
        cv2.imwrite(f'nms/pick_{i}.jpg', image)
