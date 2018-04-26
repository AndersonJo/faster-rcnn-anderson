from frcnn.preprocessing import AnchorGenerator
from frcnn.voc import PascalVocData
from tests import DATASET_ROOT_PATH
import numpy as np


def test_anchor():
    vocdata = PascalVocData(DATASET_ROOT_PATH)
    train, test, classes = vocdata.load_data()
    dataset = train + test

    anchor = AnchorGenerator(dataset)
    anchor.next_batch()
    anchor.next_batch()
