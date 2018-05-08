import os
from typing import List, Tuple

import cv2
import keras.backend as K
import numpy as np
import tensorflow as tf
from keras import Input, Model
from keras.engine import Layer
from keras.layers import TimeDistributed, Flatten, Dropout, Dense
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam

from frcnn.anchor import to_absolute_coord
from frcnn.config import Config
from frcnn.nms import non_max_suppression
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
            resized = tf.image.resize_images(fmap[:, y:y + h, x:x + w, :], (self.pool_height, self.pool_width))
            outputs.append(resized)

        final_output = K.concatenate(outputs, axis=0)
        final_output = K.reshape(final_output, (1, self.n_roi, self.pool_height, self.pool_width, self.n_channel))
        final_output = K.permute_dimensions(final_output, (0, 1, 2, 3, 4))

        return final_output

    def compute_output_shape(self, input_shape):
        return None, self.n_roi, self.pool_height, self.pool_width, self.n_channel


class ClassfierModel(object):

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
        self.n_class = len(class_mapping)  # number of classes (like car, human, bike, etc..)

        # Initiliaze Region of Interest
        self.n_roi = config.n_roi
        self.roi_pool_size = config.roi_pool_size
        self.roi_method = roi_method
        self.roi_input = Input(shape=(self.n_roi, 4))

        self.roi_cls_output = None
        self.roi_reg_output = None
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

        reg_output = TimeDistributed(Dense(4 * self.n_class, activation='linear', kernel_initializer='zero'),
                                     name='roi_regress_{}'.format(self.n_class))(h)

        self.roi_cls_output = cls_output
        self.roi_reg_output = reg_output

        img_input = self.rpn.fen.input_img
        roi_model = Model([img_input, self.roi_input], [cls_output, reg_output])
        roi_model.compile(optimizer=Adam(lr=1e-5),
                          loss=[self.clf_loss, self.regr_loss(len(self.class_mapping))])

        return roi_model

    @staticmethod
    def regr_loss(num_classes: int, huber_delta: float = 1., epsilon: float = 1e-4):
        def smooth_l1(y_true, y_pred):
            # y_true consists of two parts; labels and regressions
            # we uses only regression part
            reg_y = y_true[:, :, 4 * num_classes:]

            cond = tf.equal(reg_y, tf.constant(0.))
            cls_y = tf.where(cond, tf.zeros_like(reg_y), tf.ones_like(reg_y))

            x = K.abs(reg_y - y_pred)
            x = K.switch(x < huber_delta, 0.5 * x ** 2, x - 0.5 * huber_delta)
            loss = K.sum(x) / (K.sum(cls_y) + epsilon)

            return loss

        return smooth_l1

    @staticmethod
    def clf_loss(y_true, y_pred):
        return K.mean(categorical_crossentropy(y_true[0, :, :], y_pred[0, :, :]))

    @property
    def model(self) -> Model:
        return self.roi_model


class ClassifierNetwork(ClassfierModel):

    def non_maximum_suppression(self, rpn_cls_output: np.ndarray, rpn_reg_output: np.ndarray,
                                overlap_threshold=0.9) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param rpn_cls_output: (batch, fh, fw, 9)
        :param rpn_reg_output: (batch, fh, fw, 36) and each regression is (tx, ty, tw, th)
        :param overlap_threshold : used for Non Max Suppression
        :return
            - anchors: anchors picked by NMS. (None, (x, y, w, h))
            - probs: classification probability vector (is it object or not?)
        """
        # Transform the shape of RPN outputs to (None, 4) regression
        # Transform relative coordinates (tx, ty, tw, th) to absolute coordinates (x, y, w, h)
        anchors, probs = self._transform_rpn(rpn_reg_output, rpn_cls_output)

        # Non Maximum Suppression
        anchors, probs = non_max_suppression(anchors, probs, overlap_threshold=overlap_threshold, max_box=300)
        return anchors, probs

    def _transform_rpn(self, rpn_reg_output: np.ndarray, rpn_cls_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        The method do the followings
            1. Create Anchors on the basis of feature maps
            2. Convert reg_output (1, fh, fw, 36) to (4, fh, fw, 9) for Non Maximum Suppression
            3. Convert regression (tx, ty, tw, th) to (gx, gy, gw, gh)

        RPN Input Tensors
            - rpn_cls_output: ex. (1, 37, 50,  9) where 37 and 50 can change
            - rpn_reg_output: ex. (1, 37, 50, 36) where 37 and 50 can change

        :param rpn_cls_output: RPN classification output
        :param rpn_reg_output: RPN regression output
        :return: a tuple of anchors and classification probability
            - anchors: (min_x, min_y, max_x, max_y)
            - probs: classification probabilities (object or background)
        """
        anchor_scales = self.anchor_scales
        anchor_ratios = self.anchor_ratios
        _, fh, fw, n_anchor = rpn_cls_output.shape  # shape example (1, 37, 50, 9)

        anchors = np.zeros((4, fh, fw, n_anchor), dtype='int8')

        cur_anchor = 0
        for anchor_size in anchor_scales:
            for anchor_ratio in anchor_ratios:
                # anchor_width: Anchor's width on feature maps
                #           For example, the sub-sampling ratio of VGG-16 is 16.
                #           That is, the size of original image decrease in the ratio 6 to 1
                # anchor_height: Anchor's height
                anchor_width = (anchor_size * anchor_ratio[0]) / self.anchor_stride[0]
                anchor_height = (anchor_size * anchor_ratio[1]) / self.anchor_stride[1]

                regr = rpn_reg_output[0, :, :, cur_anchor * 4: cur_anchor * 4 + 4]  # ex. (37, 50, 4)
                regr = np.transpose(regr, (2, 0, 1))  # (4, 37, 50)

                X, Y = np.meshgrid(np.arange(fw), np.arange(fh))
                anchors[0, :, :, cur_anchor] = X  # the center coordinate of anchor's width (37, 50)
                anchors[1, :, :, cur_anchor] = Y  # the center coordinate of anchor's height (37, 50)
                anchors[2, :, :, cur_anchor] = anchor_width  # anchor width <scalar value>
                anchors[3, :, :, cur_anchor] = anchor_height  # anchor height <scalar value>
                anchors[:, :, :, cur_anchor] = to_absolute_coord(anchors[:, :, :, cur_anchor], regr)

                # it makes sure that anchors' width and height are at least 1
                # Convert (w, h) to (max_x, max_y) by adding (min_x, min_y) to (w, h)
                # anchors become (min_x, min_y, max_x, max_y) <--- Important!!!
                anchors[2, :, :, cur_anchor] = np.maximum(1, anchors[2, :, :, cur_anchor])
                anchors[3, :, :, cur_anchor] = np.maximum(1, anchors[3, :, :, cur_anchor])
                anchors[2, :, :, cur_anchor] += anchors[0, :, :, cur_anchor]
                anchors[3, :, :, cur_anchor] += anchors[1, :, :, cur_anchor]

                # Limit the anchors within the feature maps.
                anchors[0, :, :, cur_anchor] = np.maximum(0, anchors[0, :, :, cur_anchor])
                anchors[1, :, :, cur_anchor] = np.maximum(0, anchors[1, :, :, cur_anchor])
                anchors[2, :, :, cur_anchor] = np.minimum(fw - 1, anchors[2, :, :, cur_anchor])
                anchors[3, :, :, cur_anchor] = np.minimum(fh - 1, anchors[3, :, :, cur_anchor])

                cur_anchor += 1

        # A.transpose((0, 3, 1, 2)) : (4, 38 height, 50 widht, 9) -> (4, 9, 38 height, 50 width)
        # np.reshape(A.transpose((0, 3, 1, 2)), (4, -1)) : -> (4, 17100)
        anchors = np.reshape(anchors.transpose((0, 3, 1, 2)), (4, -1)).transpose((1, 0))  # (17100, 4)
        probs = rpn_cls_output.transpose((0, 3, 1, 2)).reshape((-1))  # (17100,)

        # Filter weird anchors
        min_x = anchors[:, 0]  # predicted min_x (top left x coordinate of the anchor)
        min_y = anchors[:, 1]  # predicted min_y (top left y coordinate of the anchor)
        max_x = anchors[:, 2]  # predicted max_x (bottom right x coordinate of the anchor)
        max_y = anchors[:, 3]  # predicted max_y (bottom right y coordinate of the anchor)

        idxs = np.where((min_x - max_x >= 0) | (min_y - max_y >= 0))
        anchors = np.delete(anchors, idxs, 0)
        probs = np.delete(probs, idxs, 0)

        return anchors, probs

    def debug_nms_images(self, anchors: np.ndarray, img_meta: dict):
        image = cv2.imread(img_meta['image_path'])
        image = cv2.resize(image, (img_meta['rescaled_width'], img_meta['rescaled_height']))

        ratio_x = img_meta['rescaled_width'] / img_meta['width']
        ratio_y = img_meta['rescaled_height'] / img_meta['height']

        for anchor in anchors:
            min_x = anchor[0] * 16
            min_y = anchor[1] * 16
            max_x = anchor[2] * 16 + min_x
            max_y = anchor[3] * 16 + min_y
            cx = (min_x + max_x) // 2
            cy = (min_y + max_y) // 2
            cv2.rectangle(image, (cx, cy), (cx + 5, cy + 5), (0, 0, 255))

        for obj in img_meta['objects']:
            min_x, min_y, max_x, max_y = obj[1:]
            min_x = int(min_x * ratio_x)
            max_x = int(max_x * ratio_x)
            min_y = int(min_y * ratio_y)
            max_y = int(max_y * ratio_y)

            cx = (min_x + max_x) // 2
            cy = (min_y + max_y) // 2
            cv2.rectangle(image, (cx, cy), (cx + 5, cy + 5), (255, 255, 0))

        cv2.imwrite(os.path.join('temp', img_meta['filename']), image)
