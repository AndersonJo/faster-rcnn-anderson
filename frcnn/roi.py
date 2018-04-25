from typing import List

import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.engine import Layer


class RegionOfInterestPoolingLayer(Layer):

    def __init__(self, size: List[int] = (), n_roi: int = 32, method: str = 'resize', **kwargs):
        """
        See "Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition."

        :param size: the pool size of height and width.  
        :param n_roi: the number of regions of interest  
        """
        super(RegionOfInterestPoolingLayer, self).__init__(**kwargs)
        assert len(size) == 2

        self.pool_height = size[0]
        self.pool_width = size[1]
        self.n_roi = n_roi
        self.method = method
        self.n_channel = None

    def build(self, input_shape):
        super(RegionOfInterestPoolingLayer, self).build(input_shape)
        self.n_channel = input_shape[0][-1]

    def call(self, tensors: tf.Tensor, mask=None):
        """
        :param tensors
            - tensors[0] image: the convolution features of FEN (like VGG-16) -> (batch, height, width, features)
            - tensors[1] rois: RoI Input Tensor -> (batch, number of RoI, 4)
        :param mask: ...
        """
        image = tensors[0]  # ex. (?, ?, ?, 512)
        rois = tensors[1]  # ex. (?, 32, 4)

        outputs = list()
        for roi_idx in range(self.n_roi):
            x = rois[0, roi_idx, 0]
            y = rois[0, roi_idx, 1]
            w = rois[0, roi_idx, 2]
            h = rois[0, roi_idx, 3]

            # row_length = w / float(self.pool_width)
            # col_length = h / float(self.pool_height)

            x = K.cast(x, 'int32')
            y = K.cast(y, 'int32')
            w = K.cast(w, 'int32')
            h = K.cast(h, 'int32')

            # (None, 7 pool_height, 7 pool_width, 512 n_features)
            resized = tf.image.resize_images(image[:, y:y + h, x:x + w, :], (self.pool_height, self.pool_width))
            outputs.append(resized)

        final_output = K.concatenate(outputs, axis=0)
        final_output = K.reshape(final_output, (1, self.n_roi, self.pool_height, self.pool_width, self.n_channel))
        final_output = K.permute_dimensions(final_output, (0, 1, 2, 3, 4))

        return final_output

    def compute_output_shape(self, input_shape):
        return None, self.n_roi, self.pool_height, self.pool_width, self.n_channel


def non_max_suppression_fast(boxes: np.ndarray, probs: np.ndarray = None, overlap_threshold: float = 0.3):
    """
    The code is here (https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/)
    """

    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    # Sort the bounding boxes on the basis of their scores
    if probs is not None:
        idxs = np.argsort(probs)
    else:
        idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlap_threshold)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    picked_boxes = boxes[pick].astype("int")
    if probs is None:
        return picked_boxes, None

    picked_probs = probs[pick]
    return picked_boxes, picked_probs
