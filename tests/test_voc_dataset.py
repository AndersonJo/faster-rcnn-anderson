from frcnn.voc import PascalVocData

DATASET_ROOT_PATH = '/data/VOCdevkit/'


def test_get_pascal_voc_dataset():
    voc = PascalVocData(DATASET_ROOT_PATH)
