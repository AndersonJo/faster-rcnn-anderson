from frcnn.anchor import Anchor
from frcnn.voc import PascalVocData
from tests import DATASET_ROOT_PATH


def test_anchor():
    voc = PascalVocData(DATASET_ROOT_PATH)
    train, test, classes = voc.load_data(limit_size=10)

    Anchor(test)
    import ipdb
    ipdb.set_trace()
