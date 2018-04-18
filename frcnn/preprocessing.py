import itertools
import queue
from collections import deque
from queue import Queue
from threading import Thread
from typing import List

import cv2
import numpy as np

from frcnn.config import singleton_config
from frcnn.iou import cal_iou
from frcnn.rpn import create_rpn_regression_target
from frcnn.tools import cal_rescaled_size, rescale_image, cal_fen_output_size

worker_queue = Queue()
producer_queue = Queue()


class AnchorThread(Thread):
    # Anchor Type for Regression of Region Proposal Network
    ANCHOR_NEGATIVE = 0
    ANCHOR_NEUTRAL = 1
    ANCHOR_POSITIVE = 2

    def __init__(self, receiver: Queue, sender: Queue, anchor_scales: List[int], anchor_ratios: List[float],
                 anchor_stride: List[int] = (16, 16), net_name: str = 'vgg16', rescale: bool = True,
                 overlap_min: float = 0.3, overlap_max: float = 0.6):
        super(AnchorThread, self).__init__()
        self.receiver = receiver
        self.sender = sender
        self._rescale = rescale
        self._net_name = net_name
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.anchor_stride = anchor_stride
        self.overlap_min = overlap_min
        self.overlap_max = overlap_max

    def run(self):
        while True:
            try:
                datum = self.receiver.get(timeout=10)
            except queue.Empty:
                break

            self.preprocess(datum)

            self.receiver.task_done()
            if self.receiver.empty():
                break

    def preprocess(self, datum: dict) -> dict:
        """
        The method pre-processes just a single datum (not batch data)
        :param datum: single data point (i.e. VOC Data)
        :param rescale: rescaling can improve accuracy but may decrease calculation speed in trade-off
        :return:
        """
        image = cv2.imread(datum['image_path'])
        width, height, _ = image.shape

        # Rescale Image: at least one side of image should be larger than or equal to minimum size;
        # It may improve accuracy but decrease training or inference speed in trade-off.
        if self._rescale:
            rescaled_width, rescaled_height = cal_rescaled_size(width, height)
            image = rescale_image(image, rescaled_width, rescaled_height)
        else:
            rescaled_width, rescaled_height = width, height

        datum['rescaled_width'] = rescaled_width
        datum['rescaled_height'] = rescaled_height

        self.generate_rpn_target(datum, image)

    def generate_rpn_target(self, datum: dict, image: np.ndarray):
        width = datum['width']
        height = datum['height']
        rescaled_width = datum['rescaled_width']
        rescaled_height = datum['rescaled_height']
        n_object = len(datum['objects'])
        n_ratio = len(self.anchor_ratios)
        n_anchor = len(self.anchor_ratios) * len(self.anchor_scales)

        # Calculate output size of Base Network (feature extraction model)
        output_width, output_height, _ = cal_fen_output_size(self._net_name, rescaled_width, rescaled_height)

        # Tracking best things
        best_iou_for_box = np.zeros(n_object)
        best_anchor_for_box = -1 * np.ones((n_object, 4), dtype='int')
        best_reg_for_box = np.zeros((n_object, 4), dtype='float32')
        n_pos_anchor_for_box = np.zeros(n_object)

        # Classifier Target Data
        y_cls_target = np.zeros((output_height, output_width, n_anchor))
        y_valid_box = np.zeros((output_height, output_width, n_anchor))
        y_regr_targets = np.zeros((output_height, output_width, n_anchor * 4))

        _comb = [range(output_height), range(output_width),
                 range(len(self.anchor_scales)), range(len(self.anchor_ratios)), range(n_object)]
        for y_pos, x_pos, anc_scale_idx, anc_rat_idx, idx_obj in itertools.product(*_comb):
            anc_scale = self.anchor_scales[anc_scale_idx]
            anc_rat = self.anchor_ratios[anc_rat_idx]

            # ground-truth box coordinates on the rescaled image
            obj_info = datum['objects'][idx_obj]
            gta_coord = self.cal_gta_coordinate(obj_info[1:], width, height, rescaled_width, rescaled_height)

            cv2.rectangle(image, (gta_coord[0], gta_coord[1]), (gta_coord[2], gta_coord[3]), (255, 255, 0))
            # anchor box coordinates on the rescaled image
            anchor_coord = self.cal_anchor_cooridinate(x_pos, y_pos, anc_scale, anc_rat, self.anchor_stride)

            # Check if the anchor is within the rescaled image
            _valid_anchor = self.is_anchor_valid(anchor_coord, rescaled_width, rescaled_height)
            if not _valid_anchor:
                continue

            ##############################################
            # Regression Target
            ##############################################

            # Calculate Intersection Over Union
            iou = cal_iou(gta_coord, anchor_coord)

            # Calculate regression target
            if iou > best_iou_for_box[idx_obj] or iou > self.overlap_max:
                reg_target = create_rpn_regression_target(gta_coord, anchor_coord)

            # Ground-truth bounding box should be mapped to at least one anchor box.
            # So tracking the best anchor should be implemented
            if iou > best_iou_for_box[idx_obj]:
                best_iou_for_box[idx_obj] = iou
                best_anchor_for_box[idx_obj] = (y_pos, x_pos, anc_scale_idx, anc_rat_idx)
                best_reg_for_box[idx_obj] = reg_target

            # Anchor is positive (the anchor refers to an ground-truth object) if iou > 0.5~0.7
            # is_valid_anchor: this flag prevents overwriting existing valid anchor (due to the for-loop of objects)
            #                  if the anchor meets overlap_max or overlap_min, it should not be changed.
            z_pos = anc_scale_idx + n_ratio * anc_rat_idx
            is_valid_anchor = bool(y_valid_box[y_pos, x_pos, z_pos])

            if iou > self.overlap_max:
                n_pos_anchor_for_box[idx_obj] += 1
                y_valid_box[y_pos, x_pos, z_pos] = 1
                y_cls_target[y_pos, x_pos, z_pos] = 1
                y_regr_targets[y_pos, x_pos, z_pos: z_pos + 4] = reg_target
                # self.point(image, x_pos, y_pos, (255, 255, 0))

            elif iou < self.overlap_min and not is_valid_anchor:
                y_valid_box[y_pos, x_pos, z_pos] = 1
                y_cls_target[y_pos, x_pos, z_pos] = 0
                # self.point(image, x_pos, y_pos)
            if not is_valid_anchor:
                y_valid_box[y_pos, x_pos, z_pos] = 0
                y_cls_target[y_pos, x_pos, z_pos] = 0

        y_valid_box = np.transpose(y_valid_box, (2, 0, 1))
        y_cls_target = np.transpose(y_cls_target, (2, 0, 1))

        # Ensure that at least a ground-truth bounding box is mapped to

        # for i in range(best_anchor_for_box.shape[0]):
        #     y_pos = best_anchor_for_box[i][0]
        #     x_pos = best_anchor_for_box[i][1]
        #     anc_scale = self.anchor_scales[best_anchor_for_box[i][2]]
        #     anc_rat = self.anchor_ratios[best_anchor_for_box[i][3]]
        #     self.rectangle(image, x_pos, y_pos, anc_scale, anc_rat)

        cv2.imwrite('temp/{0}.png'.format(datum['filename']), image)
        # self.sender.put('done')

    @staticmethod
    def cal_gta_coordinate(box: List[int], width: int, height: int,
                           rescaled_width: int, rescaled_height: int) -> List[int]:
        """
        The method converts coordinates to rescaled size.
        :param box: a list of coordinates [x_min, y_min, x_max, y_max]
        :param width : original width value of the image (before rescaling the image)
        :param height: original height value of the image (before rescaling the image)
        :param rescaled_height: the height of rescaled image
        :param rescaled_width: the width of rescaled image
        """

        width_ratio = rescaled_width / float(width)
        height_ratio = rescaled_height / float(height)

        x_min = round(box[0] * width_ratio)
        y_min = round(box[1] * height_ratio)
        x_max = round(box[2] * width_ratio)
        y_max = round(box[3] * height_ratio)
        return [x_min, y_min, x_max, y_max]

    @staticmethod
    def cal_anchor_cooridinate(x_pos: int, y_pos: int, anc_scale: int, anc_rat: List[float],
                               stride: List[int]) -> List[int]:
        """
        Calculates anchor coordinates on the rescaled image
        :param x_pos: x position of base network's output
        :param y_pos: y position of base network's output
        :param anc_scale: anchor size
        :param anc_rat: anchor ratios
        :param stride: anchor stride on rescaled image
        :return: a list of anchor coordinates on the rescaled image
        """
        anc_width, anc_height = anc_scale * anc_rat[0], anc_scale * anc_rat[1]

        x_min = round(stride[0] * x_pos - anc_width / 2)
        x_max = round(stride[0] * x_pos + anc_width / 2)
        y_min = round(stride[1] * y_pos - anc_height / 2)
        y_max = round(stride[1] * y_pos + anc_height / 2)

        return [x_min, y_min, x_max, y_max]

    @staticmethod
    def is_anchor_valid(anchor_coord: List[int], rescaled_width: int, rescaled_height: int) -> bool:
        """
        Check if the anchor is within the rescaled image.
        for speeding up performance, ignore anchors that are outside of the image.
        :param anchor_coord: anchor coordinates on the rescaled image
        :param rescaled_width: width of rescaled image
        :param rescaled_height: height of rescaled image
        :return: True if the anchor is within the rescaled image.
                 False if the anchor should be ignored for computation
        """
        if anchor_coord[0] < 0 or anchor_coord[2] > rescaled_width:
            return False

        if anchor_coord[1] < 0 or anchor_coord[3] > rescaled_height:
            return False
        return True

    @staticmethod
    def rectangle(image, x_pos: int, y_pos: int, anc_scale: int, anc_rat: List[float]):
        w, h = anc_scale * anc_rat[0], anc_scale * anc_rat[1]
        cv2.rectangle(image, (int(x_pos * 16 - w / 2), int(y_pos * 16 - h / 2)),
                      (int(x_pos * 16 + w / 2), int(y_pos * 16 + h / 2)),
                      (0, 0, 255))

    @staticmethod
    def point(image, x_pos: int, y_pos: int, color=(0, 0, 255)):
        cv2.rectangle(image, (x_pos * 16, y_pos * 16), (x_pos * 16 + 5, y_pos * 16 + 5), color)


class AnchorGenerator(object):
    def __init__(self, dataset: list, batch: int = 32, augment: bool = False):
        super(AnchorGenerator, self).__init__()
        self._dataset = dataset
        self.batch = batch

        self._augment = augment

        # Set Metadata
        self.n_data = len(self._dataset)

        # Set training variables
        self._cur_idx = 0

        # Get Config
        self._config = singleton_config()

        # Threads
        self.threads = list()

        # Initial Batch Queue
        self.batch_jobs = deque()
        # self.batch_jobs.append(self._put_batch_to_queue())

    def _put_batch_to_queue(self) -> int:
        if self._cur_idx >= self.n_data:
            self._cur_idx = 0

        batch_data = self._dataset[self._cur_idx:self._cur_idx + self.batch]
        n_batch = len([worker_queue.put(datum, timeout=5) for datum in batch_data])

        # Increase current index
        self._cur_idx += self.batch

        # Add batch job
        self.batch_jobs.append(n_batch)
        return n_batch

    def next_batch(self) -> List[dict]:
        # Add batch to threads
        self._put_batch_to_queue()

        # Get batch from threads
        n_batch = self.batch_jobs.pop()
        batch = list()
        try:
            for i in range(n_batch):
                batch.append(producer_queue.get(timeout=5))
        except queue.Empty as e:
            print('에러', e)
            pass

        thread_mgr = singleton_anchor_thread_manager()
        thread_mgr.restart()

        return batch


class AnchorThreadManager(object):
    def __init__(self, n_thread: int = 12):
        self.n_thread = n_thread
        self.threads = list()

    def initialize(self, start: bool = True) -> None:
        for i in range(self.n_thread):
            t = self._create_anchor_thread(start)
            self.threads.append(t)

    def _create_anchor_thread(self, start: bool = True) -> AnchorThread:
        config = singleton_config()
        t = AnchorThread(worker_queue, producer_queue, config.anchor_scales, config.anchor_ratios, config.anchor_stride,
                         config.net_name, config.is_rescale, config.overlap_max, config.overlap_min)
        t.setDaemon(True)

        if start:
            t.start()
        return t

    def restart(self):
        for i, t in enumerate(self.threads):
            if not t.isAlive():
                t = self._create_anchor_thread(True)
                self.threads[i] = t

    def wait(self):
        for t in self.threads:
            t.join()


def singleton_anchor_thread_manager() -> AnchorThreadManager:
    if hasattr(singleton_anchor_thread_manager, 'singleton'):
        return singleton_anchor_thread_manager.singleton

    config = singleton_config()
    singleton_anchor_thread_manager.singleton = AnchorThreadManager(config.n_thread)
    return singleton_anchor_thread_manager.singleton
