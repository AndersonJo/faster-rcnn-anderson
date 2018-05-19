from typing import List

import keras.backend as K
import tensorflow as tf
from keras import Input, Model
from keras.engine import Layer
from keras.layers import TimeDistributed, Flatten, Dropout, Dense
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam

from frcnn.config import Config
from frcnn.logging import get_logger
from frcnn.rpn import RegionProposalNetwork
import numpy as np

logger = get_logger(__name__)


class RegionOfInterestPoolingLayer(Layer):

    def __init__(self, size: List[int] = (), n_roi: int = 32, method: str = 'resize', **kwargs):
        """
        See "Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition."

        :param size: the pool size of height and width.  
        :param n_roi: the number of regions of interest  
        """
        super(RegionOfInterestPoolingLayer, self).__init__(**kwargs)
        assert len(size) == 2
        np.random.seed(0)

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
        fmap = tensors[0]  # ex. (?, ?, ?, 512)
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
            try:
                resized = tf.image.resize_images(fmap[:, y:y + h, x:x + w, :], (self.pool_height, self.pool_width))
                outputs.append(resized)
            except Exception as e:
                print(e)

        final_output = K.concatenate(outputs, axis=0)
        final_output = K.reshape(final_output, (1, self.n_roi, self.pool_height, self.pool_width, self.n_channel))
        final_output = K.permute_dimensions(final_output, (0, 1, 2, 3, 4))

        return final_output

    def compute_output_shape(self, input_shape):
        return None, self.n_roi, self.pool_height, self.pool_width, self.n_channel


class ClassifierNetwork(object):

    def __init__(self, rpn: RegionProposalNetwork, config: Config, class_mapping: dict, roi_method: str = 'resize'):
        """
        # Region of Interest
        :param rpn: Region Proposal Network instance
        :param config: Config instance
        :param class_mapping: dictionary of classes like {'car': 3, ...}
        :param roi_method: ('resize', 'pooling') how to do region of interest
            - resize: this method is very simple but works properly
            - pooling: slow but it does what paper says to do exactly
        """
        self.rpn = rpn

        self.anchor_scales = config.anchor_scales
        self.anchor_ratios = config.anchor_ratios
        self.anchor_stride = config.anchor_stride

        self.class_mapping = class_mapping
        self.class_mapping_rev = {v: k for k, v in class_mapping.items()}
        self.n_class = len(class_mapping)  # number of classes (like car, human, bike, etc..) including background.

        # Initiliaze Region of Interest
        self.n_roi = config.n_roi
        self.roi_pool_size = config.roi_pool_size
        self.roi_method = roi_method
        self.roi_input = Input(shape=(self.n_roi, 4))

        self.roi_cls_output = None
        self.roi_reg_output = None
        self.tensors = dict()
        self.roi_model = self._init_classifier()

    def _init_classifier(self) -> Model:
        roi_pooling_layer = RegionOfInterestPoolingLayer(size=self.roi_pool_size,
                                                         n_roi=self.n_roi,
                                                         method=self.roi_method)

        roi_pooled_output = roi_pooling_layer([self.rpn.fen.output, self.roi_input])

        h = TimeDistributed(Flatten(name='flatten'))(roi_pooled_output)
        h = TimeDistributed(Dense(4096, activation='relu', name='roi_fc1'))(h)
        h = TimeDistributed(Dropout(0.5))(h)
        h = TimeDistributed(Dense(4096, activation='relu', name='roi_fc2'))(h)
        h = TimeDistributed(Dropout(0.5))(h)

        # Classification includes background label
        cls_output = TimeDistributed(Dense(self.n_class, activation='softmax', kernel_initializer='zero'),
                                     name='roi_class_{}'.format(self.n_class))(h)

        # Exclude the background class and self.n_class has already included background class
        # You don't need coordinates if it is background.
        reg_output = TimeDistributed(Dense(4 * (self.n_class - 1), activation='linear', kernel_initializer='zero'),
                                     name='roi_regress_{}'.format(self.n_class))(h)

        self.roi_cls_output = cls_output
        self.roi_reg_output = reg_output

        self.tensors['clf_roi_pooling'] = roi_pooled_output
        self.tensors['clf_cls'] = cls_output
        self.tensors['clf_reg'] = reg_output

        image_input = self.rpn.fen.image_input
        roi_model = Model([image_input, self.roi_input], [cls_output, reg_output])
        roi_model.compile(optimizer=Adam(lr=1e-5),
                          loss=[self.clf_loss, self.regr_loss(len(self.class_mapping))])

        return roi_model

    @staticmethod
    def regr_loss(num_classes: int, huber_delta: float = 1., lambda_reg: float = 1., epsilon: float = 1e-4):
        def smooth_l1(y_true, y_pred):
            # y_true consists of two parts; labels and regressions
            # we uses only regression part
            reg_y = y_true[:, :, 4 * (num_classes - 1):]

            # cond = tf.equal(reg_y, tf.constant(0.))
            # cls_y = tf.where(cond, tf.zeros_like(reg_y), tf.ones_like(reg_y))
            cls_y = y_true[:, :, :4 * num_classes]

            x = K.abs(reg_y - y_pred)
            x = K.switch(x < huber_delta, 0.5 * x ** 2, x - 0.5 * huber_delta)
            loss = K.sum(x) / (K.sum(cls_y) + epsilon)

            return lambda_reg * loss

        def class_loss_regr_fixed_num(y_true, y_pred):
            x = y_true[:, :, 4 * (num_classes - 1):] - y_pred
            x_abs = K.abs(x)
            x_bool = K.cast(K.less_equal(x_abs, 1.0), 'float32')
            return 1 * K.sum(
                y_true[:, :, :4 * (num_classes - 1)] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(
                epsilon + y_true[:, :, :4 * (num_classes - 1)])

        return class_loss_regr_fixed_num

    @staticmethod
    def clf_loss(y_true, y_pred):
        return K.mean(categorical_crossentropy(y_true[0, :, :], y_pred[0, :, :]))

    @property
    def model(self) -> Model:
        return self.roi_model
