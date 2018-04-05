# -*- coding:utf-8 -*-
import os
from os.path import dirname
import xml.etree.ElementTree as ET


class PascalVocData(object):
    # VOC2007 uses test.txt as test data
    # VOC2017 uses val.txt as test data
    TEST_FILES = ('test.txt', 'val.txt')

    def __init__(self, voc_root_path: str, voc_names: tuple = ('VOC2007', 'VOC2012')):
        """
        :param voc_root_path: the path of "VOCdevkit" including VOC2007 or VOC2012 (i.e. '/data/VOCdevkit')
        :param voc_names: VOC challenge data (i.e. ('VOC2007', 'VOC20010', 'VOC2012'))

        voc_root_path path should look like this
        =======================================
        ├── VOC2007
        │   ├── Annotations
        │   ├── ImageSets
        │   │   ├── Layout
        │   │   ├── Main
        │   │   └── Segmentation
        │   ├── JPEGImages
        │   ├── SegmentationClass
        │   └── SegmentationObject
        ├── VOC2012
        │   ├── Annotations
        │   ├── ImageSets
        │   │   ├── Action
        │   │   ├── Layout
        │   │   ├── Main
        │   │   └── Segmentation
        │   ├── JPEGImages
        │   ├── SegmentationClass
        │   └── SegmentationObject
        =======================================
        """
        self.voc_root_path: str = voc_root_path
        self.voc_names: list = voc_names

        self._init()

    def _init(self) -> None:
        """
        Initialization method
        """
        # VOC data absolute paths
        # i.e. ['/data/VOCdevkit/VOC2007', '/data/VOCdevkit/VOC2012']
        self.voc_paths = [os.path.join(self.voc_root_path, voc_name) for voc_name in self.voc_names]
        self.voc_paths = list(filter(lambda p: os.path.exists(p), self.voc_paths))

        # Read test file names from test.txt or val.txt
        # _test_file_names: it only has list of image file names; not as absolute paths
        _test_file_names = list()
        for voc_path in self.voc_paths:
            _test = [os.path.join(voc_path, 'ImageSets', 'Main', t) for t in self.TEST_FILES]
            _test = list(filter(lambda t: os.path.exists(t), _test))
            if len(_test) <= 0:
                continue
            _test = _test[0]

            with open(_test) as f:
                for line in f:
                    _test_file_names.append(line.strip() + '.jpg')

        # Make annotation data
        train = list()
        test = list()
        for voc_path in self.voc_paths:
            _annot_dir_path = os.path.join(voc_path, 'Annotations')

            _annotation_paths = [os.path.join(_annot_dir_path, a) for a in os.listdir(_annot_dir_path)]

            for annot_path in _annotation_paths:
                annot = self.parse_annotation(annot_path)

                print(annot['image'], os.path.exists(annot['image']))
                # assert os.path.exists(annot['image'])

                if annot['filename'] in _test_file_names:
                    test.append(annot)
                else:
                    train.append(annot)

    @staticmethod
    def parse_annotation(annot_path: str) -> dict:
        """
        It parses XML VOC annotation file.
        :param annot_path: full path of annotation file
        :param validation: it should be used only for data integrity test
        :return annotation information as dictionary
        """
        annot = dict()

        # Set absolute path
        _img_dir_path = os.path.join(dirname(dirname(annot_path)), 'JPEGImages')

        # Set default information
        et: ET.ElementTree = ET.parse(annot_path)
        el: ET.ElementTree = et.getroot()

        annot['filename'] = el.find('filename').text
        annot['width'] = int(el.find('size').find('width').text)
        annot['height'] = int(el.find('size').find('height').text)
        annot['image'] = os.path.join(_img_dir_path, annot['filename'])  # absolute image path

        annot['objects'] = list()
        object_els = el.findall('object')
        for el_object in object_els:
            obj = dict()

            obj['name'] = el_object.find('name').text
            bbox = el_object.find('bndbox')
            obj['xmin'] = int(bbox.find('xmin').text)
            obj['ymin'] = int(bbox.find('ymin').text)
            obj['xmax'] = int(bbox.find('xmax').text)
            obj['ymax'] = int(bbox.find('ymax').text)
            annot['objects'].append(obj)

        return annot
