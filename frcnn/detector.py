from typing import List

import keras.backend as K
import numpy as np
import tensorflow as tf
from keras import Input
from keras.engine import Layer
from keras.layers import TimeDistributed, Flatten, Dropout, Dense

from frcnn.rpn import RegionProposalNetwork


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


class DetectionNetwork(object):

    def __init__(self, rpn: RegionProposalNetwork, n_class: int = 20,
                 n_roi: int = 32, roi_pool_size: List[int] = (7, 7), roi_method: str = 'resize'):
        """
        # Region of Interest
        :param n_class: the number of classes (PASCAL VOC is 20)
        :param n_roi: the number of regions of interest
        :param roi_pool_size: size of roi pooling layer. (height, width)
        :param roi_method: ('resize', 'pooling') how to do region of interest
            - resize: this method is very simple but works properly
            - pooling: slow but it does what paper says to do exactly
        """
        self.rpn = rpn

        # Initiliaze Region of Interest
        assert len(roi_pool_size) == 2
        self.roi_input = Input(shape=(n_roi, 4))
        self.n_class = n_class  # number of classes (like car, human, bike, etc..)
        self.n_roi = n_roi
        self.roi_method = roi_method
        self.pool_height = roi_pool_size[0]
        self.pool_widht = roi_pool_size[1]
        self.roi_cls_output = None
        self.roi_reg_output = None
        self.roi_pooling_layer = RegionOfInterestPoolingLayer(size=roi_pool_size, n_roi=n_roi, method=roi_method)
        self._init_classifier()

    def _init_classifier(self) -> List[np.ndarray]:
        roi_pooled_output = self.roi_pooling_layer([self.rpn.fen.output, self.roi_input])

        h = TimeDistributed(Flatten(name='flatten'))(roi_pooled_output)
        h = TimeDistributed(Dense(4096, activation='relu', name='roi_fc1'))(h)
        h = TimeDistributed(Dropout(0.5))(h)
        h = TimeDistributed(Dense(4096, activation='relu', name='roi_fc2'))(h)
        h = TimeDistributed(Dropout(0.5))(h)

        cls_output = TimeDistributed(Dense(self.n_class, activation='softmax', kernel_initializer='zero'),
                                     name='roi_class_{}'.format(self.n_class))(h)

        reg_output = TimeDistributed(Dense(4 * (self.n_class - 1), activation='linear', kernel_initializer='zero'),
                                     name='roi_regress_{}'.format(self.n_class))(h)

        self.roi_cls_output = cls_output
        self.roi_reg_output = reg_output

        return [cls_output, reg_output]
