from queue import Queue
from threading import Thread
from time import sleep
from typing import List

import cv2

from frcnn.config import singleton_config
from frcnn.tools import cal_rescaled_size, rescale_image, cal_fen_output_size

receiver_queue = Queue()
sender_queue = Queue()


class AnchorThreadManager(object):
    def __init__(self, n_thread: int = 12):
        self.n_thread = n_thread
        self.threads = list()

    def create_anchor_threads(self):
        for i in range(self.n_thread):
            t = AnchorThread(receiver_queue)
            t.setDaemon(True)
            self.threads.append(t)
            t.start()


class AnchorThread(Thread):
    def __init__(self, receiver: Queue):
        super(AnchorThread, self).__init__()
        self.receiver = receiver

    def run(self):
        while True:
            datum = self.receiver.get()
            raise Exception

            self.receiver.task_done()

    def generate_rpn_target(self, resized_width: int, resized_height: int):
        # Calculate output size of Base Network (feature extraction model)
        output_width, output_height, _ = cal_fen_output_size(self._config.net_name, resized_width, resized_height)

        # Response variables

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
            resized_width, resized_height = cal_rescaled_size(width, height)
            image = rescale_image(image, resized_width, resized_height)
        else:
            resized_width, resized_height = width, height

        self.generate_rpn_target(resized_width, resized_height)


# TODO: Anchor안에는 Region Proposal의 target data를 생성해주는 메소드가 존재한다.
# TODO: 그렇다고 anchor라는 개념이 없는것도 아니다. 서로 얽히고 섥혀있어 논리적 클래스의 구조를 어떻게 잡을지가 문제다.
class Anchor(object):
    def __init__(self, dataset: list, batch: int = 32, rescale: bool = True, augment: bool = False):
        super(Anchor, self).__init__()
        self._dataset = dataset
        self.batch = batch
        self._rescale = rescale
        self._augment = augment

        # Set Metadata
        self.n_data = len(self._dataset)

        # Set training variables
        self._cur_idx = 0

        # Get Config
        self._config = singleton_config()

        # Threads
        self.threads = list()

    def create_receiver_threads(self):

    def next_batch(self) -> List[dict]:
        if self._cur_idx >= self.n_data:
            self._cur_idx = 0

        batch_data = self._dataset[self._cur_idx:self._cur_idx + self.batch]
        batch_data = [receiver_queue.put(datum) for datum in batch_data]

        # Increase current index
        self._cur_idx += self.batch

        sleep(10)

        return batch_data
