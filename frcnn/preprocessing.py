import itertools
import queue
from collections import deque
from queue import Queue
from threading import Thread
from typing import List

import cv2
import numpy as np

from frcnn.config import singleton_config, Config
from frcnn.tools import cal_rescaled_size, rescale_image, cal_fen_output_size, cal_iou

worker_queue = Queue()
producer_queue = Queue()


class AnchorThreadManager(object):
    def __init__(self, n_thread: int = 12):
        self.n_thread = n_thread
        self.threads = list()

    def create_anchor_threads(self, start: bool = True) -> None:
        config = singleton_config()
        for i in range(self.n_thread):
            t = AnchorThread(config, worker_queue)
            t.setDaemon(True)
            self.threads.append(t)
            if start:
                t.start()


class AnchorThread(Thread):
    def __init__(self, config: Config, receiver: Queue):
        super(AnchorThread, self).__init__()
        self.receiver = receiver
        self._config = config
        self._rescale = config.is_rescale

    def run(self):
        while True:
            datum = self.receiver.get()
            self.preprocess(datum)

            self.receiver.task_done()

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
        anchor_stride = self._config.anchor_stride

        # Calculate output size of Base Network (feature extraction model)
        output_width, output_height, _ = cal_fen_output_size(self._config.net_name, rescaled_width, rescaled_height)

        # Response variables
        _comb = [range(n_object), self._config.anchor_scales, self._config.anchor_ratios,
                 range(output_width), range(output_height)]
        for idx_obj, anc_scale, anc_rat, x_pos, y_pos in itertools.product(*_comb):
            # ground-truth box coordinates on the rescaled image
            obj_info = datum['objects'][idx_obj]
            gta_coord = self.cal_gta_coordinate(obj_info[1:], width, height, rescaled_width, rescaled_height)

            # anchor box coordinates on the rescaled image
            anchor_coord = self.cal_anchor_cooridinate(x_pos, y_pos, anc_scale, anc_rat, anchor_stride)

            # Check if the anchor is within the rescaled image
            _valid_anchor = self.is_anchor_valid(anchor_coord, rescaled_width, rescaled_height)
            if not _valid_anchor:
                continue

            # Calculate Intersection Over Union
            iou = cal_iou(gta_coord, anchor_coord)
            if iou > 0.5:
                print(iou)

            # print(datum)
            # print(gta_coord[0:2], gta_coord[2:])
            # cv2.rectangle(image, tuple(gta_coord[0:2]), tuple(gta_coord[2:]), (0, 0, 255))
            cv2.rectangle(image, (anchor_coord[0], anchor_coord[1]), (anchor_coord[2], anchor_coord[3]), (0, 0, 255))
        cv2.imwrite('temp/{0}.png'.format(datum['filename']), image)

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


class Anchor(object):
    def __init__(self, dataset: list, batch: int = 32, augment: bool = False):
        super(Anchor, self).__init__()
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
        self.batch_jobs.append(self._put_batch_to_queue())

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

        # print(batch)
        # print()
        return batch


def singleton_anchor_thread_manager() -> AnchorThreadManager:
    if hasattr(singleton_anchor_thread_manager, 'singleton'):
        return singleton_anchor_thread_manager.singleton

    config = singleton_config()
    singleton_anchor_thread_manager.singleton = AnchorThreadManager(config.n_thread)
    return singleton_anchor_thread_manager.singleton
