import itertools
import queue
from collections import deque
from queue import Queue
from threading import Thread
from typing import List

import cv2
import numpy as np

from frcnn.config import singleton_config, Config
from frcnn.tools import cal_rescaled_size, rescale_image, cal_fen_output_size

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
        rescaled_width = datum['rescaled_width']
        rescaled_height = datum['rescaled_height']
        n_object = len(datum['objects'])

        # Calculate output size of Base Network (feature extraction model)
        output_width, output_height, _ = cal_fen_output_size(self._config.net_name, rescaled_width, rescaled_height)

        # Response variables
        _comb = [self._config.anchor_scales, self._config.anchor_ratios,
                 range(output_width), range(output_height), range(n_object)]
        for anc_scale, anc_rat, out_width, out_height, idx_obj in itertools.product(*_comb):
            pass

        print(datum)


# TODO: Anchor안에는 Region Proposal의 target data를 생성해주는 메소드가 존재한다.
# TODO: 그렇다고 anchor라는 개념이 없는것도 아니다. 서로 얽히고 섥혀있어 논리적 클래스의 구조를 어떻게 잡을지가 문제다.
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

        print(batch)
        return batch


def singleton_anchor_thread_manager() -> AnchorThreadManager:
    if hasattr(singleton_anchor_thread_manager, 'singleton'):
        return singleton_anchor_thread_manager.singleton

    config = singleton_config()
    singleton_anchor_thread_manager.singleton = AnchorThreadManager(config.n_thread)
    return singleton_anchor_thread_manager.singleton
